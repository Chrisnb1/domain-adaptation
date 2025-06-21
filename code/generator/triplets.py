import json
import random
import time

class TripletDatasetGenerator:
    def load_json_data(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as infile:
            return [json.loads(line) for line in infile]
          
    def get_negative_context(self, positive: str, contexts) -> str:
        other_contexts = [ctx for ctx in contexts if ctx != positive]
        return random.choice(other_contexts)
    
    def build_triplets(self, input_path, output_path):
        start_time = time.time()
        processed_count = 0
        data = self.load_json_data(input_path)
        contexts = [context['context'] for context in data]

        print(f"Iniciando procesamiento de {input_path}")
        print(f"Salida en: {output_path}")

        with open(output_path, 'w', encoding="utf-8") as outfile:
            for entry in data:
                positive = entry['context']
                negative = self.get_negative_context(positive, contexts)
                for query in entry['queries'].values():
                    triplet_obj = {
                        'query': query,
                        'positive': positive,
                        'negative': negative
                    }
                    outfile.write(json.dumps(triplet_obj, ensure_ascii=False) + "\n")
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        print(f"Cantidad de 'triplet' generados: {processed_count}")
                    
                        

        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n=== RESUMEN ===")
        print(f"Líneas procesadas: {processed_count}")
        print(f"Tiempo total: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

if __name__ == "__main__":
    input_path = '../corpus/queries_gemma.jsonl'
    output_path = '../corpus/triplets_gemma.jsonl'

    triplets = TripletDatasetGenerator()
    triplets.build_triplets(input_path, output_path)
