import json
import random
from faker import Faker

fake = Faker()

def noisy_stt(text, entity_spans):
    noisy = text.lower()
    tokens = noisy.split(' ')
    replacements = {".": " dot", "@": " at ", ",": "", "?": "", "!": "", "-": " "}
    for k, v in replacements.items():
        noisy = noisy.replace(k, v)
    noisy = " ".join(noisy.split())
    
    new_entities = []
    for ent in entity_spans:
        val = ent['value'].lower()
        for k, v in replacements.items():
            val = val.replace(k, v)
        val = " ".join(val.split())
        start = noisy.find(val)
        if start != -1:
            new_entities.append({"start": start, "end": start + len(val), "label": ent['label']})
            
    return noisy, new_entities

def generate_dataset(num_samples, filename):
    data = []
    for i in range(num_samples):
        templates = [
            ("my email is {val}", "EMAIL", fake.email),
            ("contact me at {val}", "PHONE", fake.phone_number),
            ("i live in {val}", "CITY", fake.city),
            ("my name is {val}", "PERSON_NAME", fake.name),
            ("credit card number is {val}", "CREDIT_CARD", fake.credit_card_number),
            ("today is {val}", "DATE", fake.date)
        ]
        tmpl, label, func = random.choice(templates)
        val = func()
        text = tmpl.format(val=val)
        entities = [{"start": 0, "end": 0, "label": label, "value": val}]
        
        final_text, final_entities = noisy_stt(text, entities)
        if final_entities:
            entry = {"id": f"utt_{i:04d}", "text": final_text, "entities": final_entities}
            data.append(entry)

    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Generated {len(data)} samples in {filename}")

if __name__ == "__main__":
    generate_dataset(800, 'data/train.jsonl')
    generate_dataset(200, 'data/dev.jsonl')
