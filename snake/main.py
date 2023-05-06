import os
import glob
import hashlib
import json
import textwrap
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
        # Calculate the dot product of vectors a and b
        # Multiply each item in array 'a' with the corresponding item in array 'b' and then sum them
        dot_product = sum(ai * bi for ai, bi in zip(a, b))

        # Calculate the magnitude (Euclidean norm) of vector a
        # Square each item in array 'a', sum the squared values, and then take the square root of the sum
        magnitude_a = (sum(ai ** 2 for ai in a)) ** 0.5

        # Calculate the magnitude (Euclidean norm) of vector b
        # Square each item in array 'b', sum the squared values, and then take the square root of the sum
        magnitude_b = (sum(bi ** 2 for bi in b)) ** 0.5

        # Calculate the cosine similarity as the dot product divided by the product of magnitudes
        # Divide the dot product by the product of the magnitudes of vectors 'a' and 'b'
        return dot_product / (magnitude_a * magnitude_b)
    except Exception as e:
        print('Error computing cosine similarity:', e)
        return -1


# Example usage:
a = [1, 2, 3]
b = [4, 5, 6]

similarity = cosine_similarity(a, b)
print('Cosine similarity:', similarity)


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
