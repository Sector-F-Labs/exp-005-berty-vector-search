import os
import glob
import hashlib
import json
import textwrap
import numpy as np
import redis
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
from torch import tensor


def read_text_files(directory):
    texts = []
    for file in glob.glob(os.path.join(directory, '*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    print('Read', len(texts), 'documents')
    return texts


def create_bert_embeddings(model, texts):
    tokenizer = AutoTokenizer.from_pretrained(model)
    bert_model = AutoModel.from_pretrained(model)

    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt',
                           truncation=True, padding=True)
        outputs = bert_model(**inputs)
        embeddings.append({
            "values": normalize(outputs.last_hidden_state.mean(
                dim=1), p=2).detach().numpy().tolist()[0]
        })

    return embeddings


def store_embeddings_in_redis(redis_connection, texts, embeddings):
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        key = f'embedding:{hash}'
        embedding_json = json.dumps(embedding)
        redis_connection.hset(key, 'text', text)
        redis_connection.hset(key, 'embedding', embedding_json)
        print(f'Stored embedding for document {i} with key {key}')


def cosine_similarity(a, b):
    try:
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        return dot_product / (magnitude_a * magnitude_b)
    except Exception as e:
        print('Error computing cosine similarity:', e)
        return -1


def retrieve_documents_and_compute_similarity(redis_connection, query_embedding):
    document_keys = redis_connection.keys('embedding:*')
    document_similarity_scores = []

    for key in document_keys:
        embedding_json = redis_connection.hget(key, 'embedding')
        document_embedding = json.loads(embedding_json)

        similarity = cosine_similarity(
            query_embedding['values'], document_embedding['values'])
        text = redis_connection.hget(key, 'text')
        document_similarity_scores.append((text, similarity))

    return document_similarity_scores


def choose_best_answer(scores: list[tuple[str, float]]):
    best_score = -1
    best_answer = None
    for text, score in scores:
        if score > best_score:
            best_score = score
            best_answer = text
    return best_answer


def main():
    directory = 'texts'
    texts = read_text_files(directory)

    model = 'sentence-transformers/all-MiniLM-L12-v2'
    try:
        embeddings = create_bert_embeddings(model, texts)
    except Exception as e:
        print('Error creating embeddings:', e)

    redis_url = 'redis://127.0.0.1/'
    r = redis.StrictRedis.from_url(redis_url)

    try:
        store_embeddings_in_redis(r, texts, embeddings)
    except Exception as e:
        print('Error storing embeddings in Redis:', e)

    query = 'TypeScript code'
    print(f'\nSearching docs for {query}...')
    query_embeddings = create_bert_embeddings(model, [query])
    query_embedding = query_embeddings[0]

    scores = retrieve_documents_and_compute_similarity(r, query_embedding)

    best_answer = choose_best_answer(scores)
    decoded = best_answer.decode('utf-8')
    print(f'found:\n{decoded}')


if __name__ == '__main__':
    main()
