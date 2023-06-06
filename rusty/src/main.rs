use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

use rust_bert::RustBertError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::Path;

use redis::Commands;
use serde_json;

mod compare;

use crate::compare::cosine_similarity;

#[macro_use]
extern crate rustacuda;

#[macro_use]
extern crate rustacuda_derive;

extern crate rustacuda_core;


fn read_text_files(directory: &Path) -> Vec<String> {
    let mut texts = Vec::new();
    if directory.is_dir() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.extension().unwrap() == "txt" {
                let mut text = String::new();
                let mut file = fs::File::open(path).unwrap();
                file.read_to_string(&mut text).unwrap();
                texts.push(text);
            }
        }
    }
    texts
}

#[derive(Debug, Serialize, Deserialize)]
struct BERTEmbedding {
    pub values: Vec<f32>,
}

fn create_bert_embeddings(texts: &[String]) -> Result<Vec<BERTEmbedding>, RustBertError> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let embeddings = model.encode(texts)?;

    let ebd = embeddings
        .into_iter()
        .map(|e| BERTEmbedding { values: e })
        .collect();

    Ok(ebd)
}

fn store_embeddings_in_redis(
    redis_connection: &mut redis::Connection,
    texts: &[String],
    embeddinbgs: &[BERTEmbedding],
) -> redis::RedisResult<()> {
    for (i, (text, embedding)) in texts.iter().zip(embeddinbgs).enumerate() {
        //md5 hash of the text
        let hash = md5::compute(text);

        let key = format!("embedding:{:x}", hash);
        let embedding_json = serde_json::to_string(embedding).unwrap();
        let _: () = redis_connection.hset(&key, "text", text)?;
        let _: () = redis_connection.hset(&key, "embedding", &embedding_json)?;

        println!("Stored embedding for document {} with key {}", i, key);
    }
    Ok(())
}


fn retrieve_documents_and_compute_similarity(
    conn: &mut redis::Connection,
    query_embedding: &BERTEmbedding,
) -> redis::RedisResult<Vec<(String, f32)>> {
    let document_keys: Vec<String> = conn.keys("embedding:*")?;
    println!("document_keys: {:?}", document_keys);
    let mut document_similarity_scores: Vec<(String, f32)> = Vec::new();

    for key in document_keys {
        let document: BERTEmbedding = {
            let embedding_json: String = conn.hget(&key, "embedding")?;
            serde_json::from_str(&embedding_json).unwrap()
        };
        let similarity = cosine_similarity(&query_embedding.values, &document.values);
        let text: String = conn.hget(&key, "text")?;
        document_similarity_scores.push((text, similarity));
    }

    Ok(document_similarity_scores)
}

fn main() {
    // Result<(), Error> {
    const URL: &str = "redis://127.0.0.1/";

    // Read the text files
    let directory = Path::new("./texts");
    let texts = read_text_files(&directory);

    let embeddings = match create_bert_embeddings(texts.as_slice()) {
        Ok(e) => e,
        Err(e) => panic!("Error creating embeddings: {}", e),
    };

    // Connect to Redis
    let client = match redis::Client::open(URL) {
        Ok(c) => c,
        Err(e) => panic!("Error opening Redis: {}", e),
    };

    let mut redis_connection = match client.get_connection() {
        Ok(c) => c,
        Err(e) => panic!("Error connecting to Redis: {}", e),
    };

    match store_embeddings_in_redis(&mut redis_connection, &texts, &embeddings) {
        Ok(_) => println!("Stored embeddings in Redis"),
        Err(e) => println!("Error storing embeddings in Redis: {}", e),
    }

    let query = "console.log()";
    let query_embeddings = create_bert_embeddings(&[query.to_owned()]).unwrap();
    let query_embedding = &query_embeddings[0];

    let scores =
        retrieve_documents_and_compute_similarity(&mut redis_connection, query_embedding).unwrap();

    println!("scores: {:?}", scores);
    // Ok(())
}
