import yaml
import json
import time
import re
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class SyntheticDatasetGenerator:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.system_prompt = self.config['question_generation.system_prompt']
        self.health_keywords = self.config['question_generation.health_keywords']
        self.min_words = self.config.get('question_generation.min_words', 15)
        self.min_chars = self.config.get('question_generation.min_chars', 100)
        self.llm = self.load_llm()

    def load_config(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file_config:
            return yaml.safe_load(file_config)
        
    def load_llm(self):
        generative_model = self.config['generative_model']
        return ChatOllama(
                    model=generative_model['model_name'],
                    temperature=generative_model['temperature'], 
                    num_predict=generative_model['num_predict'],  
                    repeat_penalty=generative_model['repeat_penalty'], 
                    top_k=generative_model['top_k'], 
                    top_p=generative_model['top_p']
                )
    
    def evaluate_health_related(self, text):
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.health_keywords)
    
    def evaluate_len_content(self, text):
        clean_text = re.sub(r'\s+', ' ', text.strip())
                
        word_count = len(clean_text.split())
        char_count = len(clean_text)
        
        return word_count >= self.min_words and char_count >= self.min_chars
    
    def create_prompt_template(self, context):
        prompt_template = f"""
        {self.system_prompt}

        **IMPORTANTE**: 
        - Responde EXACTAMENTE en el formato JSON solicitado
        - Si el contexto NO es relevante o es insuficiente, responde únicamente: "Este tema no es relevante."
        - NO agregues explicaciones adicionales
        - Las preguntas deben ser específicas y basadas directamente en el contexto proporcionado

        **Contexto a evaluar:** {context}

        **Tu respuesta debe ser:**
        - Si es relevante: JSON con las 3 preguntas
        - Si NO es relevante: "Este tema no es relevante."
        """
        return prompt_template
    
    def invoke_model(self, context, max_retries=2):
        if not self.evaluate_health_related(context):
            return "Este tema no es relevante."
        
        if not self.evaluate_len_content(context):
            return "Este tema no es relevante."
        
        system_prompt_template = self.create_prompt_template(context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("human", "Evalúa el contexto y genera la respuesta según las instrucciones.")
        ])
        
        chain = prompt | self.llm
        
        for attempt in range(max_retries + 1):
            try:
                response = chain.invoke({"context": context})
                content = response.content.strip()
                
                if self.validate_response(content):
                    return content
                else:
                    print(f"Respuesta inválida en intento {attempt + 1}: {content[:100]}...")
                    
            except Exception as e:
                print(f"Error en intento {attempt + 1}: {e}")
                if attempt == max_retries:
                    return "Este tema no es relevante."
                time.sleep(1) 
        
        return "Este tema no es relevante."
    
    def validate_response(self, response):
        if "Este tema no es relevante." in response:
            return True
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                required_keys = ['question_1', 'question_2', 'question_3']
                return all(key in json_data for key in required_keys)
        except:
            pass
        
        return False
    
    def extract_json_from_response(self, response):
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                queries = json.loads(json_str)
                
                required_keys = ['question_1', 'question_2', 'question_3']
                if all(key in queries for key in required_keys):
                    if all(len(str(queries[key]).strip()) > 10 for key in required_keys):
                        return queries
            
            return None
            
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON: {e}")
            print(f"Respuesta problemática: {response[:200]}...")
            return None
    
    def process_line(self, line):
        line = line.strip()
        if not line:
            return None
        
        response = self.invoke_model(line)
        
        if "Este tema no es relevante." not in response:
            queries = self.extract_json_from_response(response)
            if queries:
                return queries
        
        return None
    
    def build_synthetic_dataset(self, input_path, output_path, max_lines=None):
        start_time = time.time()
        processed_count = 0
        valid_count = 0
        
        print(f"Iniciando procesamiento de {input_path}")
        print(f"Salida en: {output_path}")
        
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            context_id = 1
            
            for idx, line in enumerate(infile):
                if max_lines and idx >= max_lines:
                    break
                
                processed_count += 1
                queries = self.process_line(line)
                
                if queries is not None:
                    json_record = {
                        "context_id": context_id,
                        "context": line.strip(),
                        "queries": queries
                    }
                    
                    outfile.write(json.dumps(json_record, ensure_ascii=False) + "\n")
                    print(f"Contexto {context_id}: Preguntas generadas")
                    context_id += 1
                    valid_count += 1
                
                # Log de progreso
                if idx % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(f"Progreso: {processed_count} procesadas, {valid_count} válidas "
                          f"({rate:.1f} líneas/seg)")
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        success_rate = (valid_count / processed_count) * 100 if processed_count > 0 else 0
        
        print(f"\n=== RESUMEN ===")
        print(f"Líneas procesadas: {processed_count}")
        print(f"Contextos válidos: {valid_count}")
        print(f"Tasa de éxito: {success_rate:.1f}%")
        print(f"Tiempo total: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
        print(f"Promedio: {elapsed_time/processed_count:.2f} seg/línea")

if __name__ == "__main__":
    try:
        config_file = 'config.yaml'
        generator = SyntheticDatasetGenerator(config_file)
        print("Generador cargado correctamente")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    input_path = '../corpus/cowese_short.txt'
    output_path = '../corpus/queries_gemma.jsonl'
    
    generator.build_synthetic_dataset(
        input_path, 
        output_path, 
        max_lines=None  # Limitar para pruebas, None para procesar todo
    )

