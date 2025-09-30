
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import json
from typing import Dict, Any, List

import torch
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# Carrega .env se existir
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)
    
    # Verifica se o oráculo está disponível
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("🔮 Oráculo OpenAI ativo - feedback será gerado durante o treinamento")
    else:
        logger.warning("⚠️  Oráculo OpenAI inativo - OPENAI_API_KEY não configurada")
        logger.warning("   Para ativar: export OPENAI_API_KEY='sua_chave_aqui'")

    # Funções do oráculo definidas após carregamento do modelo para acesso correto
    def call_oracle_llm(response: str, solution: str = None) -> str:
        """
        Chama o LLM oráculo para analisar a resposta e gerar log de erro.
        Usa a solução correta para gerar feedback mais preciso.
        Usa variável de ambiente para a API key por segurança.
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
                logger.warning("Para usar o oráculo, defina: export OPENAI_API_KEY='sua_chave_aqui'")
                logger.warning("Continuando sem feedback do oráculo...")
                return "Feedback do oráculo não disponível (API key não configurada)."
            
            client = OpenAI(api_key=api_key)
            MODEL = "gpt-4o-mini"
            logger.debug(f"Oráculo ativo: usando {MODEL} para feedback")
            
            # Constrói o prompt incluindo a solução correta
            if solution:
                prompt = f"""ANALYSIS INSTRUCTIONS:

                    Compare the agent's response with the expected solution and provide constructive feedback.

                    EXPECTED SOLUTION:
                    {solution}

                    AGENT'S RESPONSE:
                    {response}

                    FEEDBACK GUIDELINES:

                    Evaluate whether the response is correct or incorrect

                    If incorrect: specifically identify what is wrong (incorrect action, invalid parameters, flawed logic)

                    If correct: confirm success

                    Be concise and specific in comments

                    Focus only on aspects that need improvement or correction

                    Use the provided examples as format reference

                    Use only 150 characters to provide the feedback.

                    FEEDBACK EXAMPLES:

                    "Incorrect: get_order_details cannot be called for this problem"

                    "Correct: Action successful - get_reservation_details was appropriate"

                    "Incorrect: Incorrect parameter, reservation_id should be numeric, not alphanumeric"

                    "Correct: Logical sequence correct and authentication adequate"

                    "Incorrect: Json is not valid for to call the API"

                    PROVIDE FEEDBACK:
                """
            else:
                prompt = f"If there is no solution, just say that it is not possible to analyze the response."

            response_api = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": """
                    You are a specialist who analyzes agents' responses and provides accurate feedback by 
                    comparing them to correct solutions. 
                    Provide concise, specific, and helpful error logs/feedback to improve the response.
                    """},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            return response_api.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling oracle: {e}")
            return "Error accessing oracle."

    def make_conversation(example: Dict[str, Any], prompt_column: str = script_args.dataset_prompt_column) -> Dict[str, Any]:
        """
        Constrói o prompt inicial do dataset seguindo a estrutura:
        Compatível com ambos formatos:
        1. Dataset simples: system_prompt do config + prompt_column
        2. Dataset complexo: JSON com system + tools + prompt_turn_1 + prompt_turn_2
        """
        prompt = []
        
        # Primeiro tenta o formato complexo (JSON)
        problem_data = example.get(prompt_column, None)
        if problem_data is None:
            raise ValueError(f"Exemplo sem coluna '{prompt_column}'.")
        
        # Verifica se é formato JSON complexo
        is_complex_format = False
        try:
            if isinstance(problem_data, str):
                # Tenta parsear como JSON
                parsed_data = json.loads(problem_data)
                if isinstance(parsed_data, dict) and ('system' in parsed_data or 'tools' in parsed_data):
                    problem_data = parsed_data
                    is_complex_format = True
            elif isinstance(problem_data, dict) and ('system' in problem_data or 'tools' in problem_data):
                is_complex_format = True
        except json.JSONDecodeError:
            # Não é JSON válido, usa formato simples
            is_complex_format = False
        
        if is_complex_format:
            logger.debug("Usando formato complexo (JSON com system/tools)")
            
            # Adiciona system + tools com tratamento robusto de tipos
            system_content = problem_data.get('system', '')
            tools_content = problem_data.get('tools', '')
            
            # Garante que system_content seja string
            if system_content is None:
                system_content = ''
            else:
                system_content = str(system_content)
            
            # Converte tools para string se for lista ou outro tipo
            if isinstance(tools_content, list):
                tools_content = '\n'.join(str(tool) for tool in tools_content)
            elif tools_content is None:
                tools_content = ''
            else:
                tools_content = str(tools_content)
            
            system_full = system_content
            if tools_content:
                system_full += '\n\nTOOLS:\n' + tools_content
            
            prompt.append({"role": "system", "content": system_full})

            # Adiciona os turnos do usuário com tratamento de tipos
            turn_1 = problem_data.get('prompt_turn_1', '')
            turn_2 = problem_data.get('prompt_turn_2', '')
            
            # Garante que os turnos sejam strings
            if turn_1 is not None:
                turn_1 = str(turn_1)
            else:
                turn_1 = ''
                
            if turn_2 is not None:
                turn_2 = str(turn_2)
            else:
                turn_2 = ''
            
            if turn_1:
                prompt.append({"role": "user", "content": turn_1})
            if turn_2:
                prompt.append({"role": "user", "content": turn_2})
        
        else:
            logger.debug("Usando formato simples (system_prompt do config + prompt_column)")
            
            # Formato simples - usa system_prompt do config
            if training_args.system_prompt is not None:
                prompt.append({"role": "system", "content": training_args.system_prompt})

            # Adiciona o conteúdo da coluna como prompt do usuário
            if isinstance(problem_data, str):
                user_content = problem_data
            else:
                user_content = str(problem_data)
            
            prompt.append({"role": "user", "content": user_content})

        return {"prompt": prompt}


    def generate_response_with_model(prompt_messages: List[Dict[str, str]], model_ref, tokenizer_ref, max_tokens: int = 256) -> str:
        """
        Gera resposta usando o modelo carregado.
        Esta função simula a geração que seria feita pelo GRPOTrainer.
        """
        try:
            # Converte messages para texto simples para geração
            prompt_text = ""
            for msg in prompt_messages:
                if msg["role"] == "system":
                    prompt_text += f"Sistema: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt_text += f"Usuário: {msg['content']}\n\n"
            
            prompt_text += "Assistente: "
            
            # Tokeniza e gera
            inputs = tokenizer_ref(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
            
            # Detecta o dispositivo do modelo de forma robusta
            try:
                device = next(model_ref.parameters()).device
            except StopIteration:
                # Fallback se o modelo não tiver parâmetros
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Move todos os tensors para o dispositivo correto
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Garante que o modelo está no mesmo dispositivo
            model_ref = model_ref.to(device)
            
            with torch.no_grad():
                outputs = model_ref.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer_ref.eos_token_id,
                    use_cache=True
                )
            
            # Decodifica apenas os tokens gerados
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer_ref.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return "Erro na geração da resposta."

    def process_with_oracle(example: Dict[str, Any], model_ref=None, tokenizer_ref=None) -> Dict[str, Any]:
        """
        Pipeline completo: geração inicial → oráculo com solução → geração final
        """
        # Monta prompt inicial
        conv = make_conversation(example)
        prompt = conv["prompt"].copy()  # Copia para não modificar o original
        
        # Obtém a solução correta do exemplo
        solution = example.get("solution", None)

        # 1. Gera resposta inicial do LLM
        initial_response = generate_response_with_model(prompt, model_ref, tokenizer_ref)

        # 2. Chama o oráculo passando a solução para comparação
        oracle_feedback = call_oracle_llm(initial_response, solution)

        # 3. Incrementa o system prompt com o feedback do oráculo
        prompt_with_oracle = prompt.copy()
        for msg in prompt_with_oracle:
            if msg["role"] == "system":
                msg["content"] += f"\n\n[FEEDBACK DO ORÁCULO]: {oracle_feedback}"
                break

        # 4. Gera nova resposta com o feedback do oráculo
        final_response = generate_response_with_model(prompt_with_oracle, model_ref, tokenizer_ref)

        # Retorna no formato esperado pelo GRPOTrainer
        return {
            "prompt": prompt,  # Prompt original para o GRPOTrainer
            "response": final_response,  # Resposta final após feedback do oráculo
            "solution": solution,  # Solução correta
            # Campos adicionais para logging/debug
            "initial_response": initial_response,
            "oracle_feedback": oracle_feedback
        }

    # Aplica o pipeline oráculo ao dataset
    logger.info("*** Aplicando pipeline oráculo ao dataset ***")
    
    # Cria função com closure para capturar model e tokenizer
    def process_example_with_oracle(example):
        return process_with_oracle(example, model, tokenizer)
    
    dataset = dataset.map(process_example_with_oracle, desc="Processando com oráculo")

    # Remove colunas desnecessárias
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    # O dataset já está no formato correto após process_with_oracle
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
