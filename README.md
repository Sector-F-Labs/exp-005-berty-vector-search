# exp-005-berty-vector-search
Experiments to see what would be the easiest way to create text embeddings and then use vector search for enriching ChatGPT Prompts.

## Goal

The goal of this experiment is to compare the difficulty of using BERT embeddings with different database stacks for vector storage and find out which language has simpler support while being as local as possible.

Rust has the benefit of being able to tap into local models through llama-rs and rust-bert, but Go is also a viable language for this kind of task.

## Method

The method of this experiment is to create a simple application that uses BERT to encode text files and store them in a vector database. The application will then search for the vector that best matches a user query. Difficulties will be noted and compared, and a conclusion will be drawn.

## Findings for Rust

Rust provides support for BERT models through the `rust-bert` library, which allows easy interaction with pre-trained models. However, the library is still maturing and may have some limitations compared to more established options like the Hugging Face Transformers library in Python. Working with Redis in Rust is straightforward using the `redis` crate.

```rust
// Add your Rust findings here.
```

## Findings for Go

Go has a well-supported library for interacting with Redis, called `redisearch-go`. While there are no mature libraries for working with BERT models directly in Go, you can use TensorFlow or ONNX to load and use pre-trained BERT models. Alternatively, you can use an external service for generating embeddings.

```go
// Add your Go findings here.
```

## Conclusion

Based on the findings, Rust and Go both have their advantages and disadvantages when working with BERT embeddings and vector databases. Rust has better support for BERT models, while Go has more mature libraries for database interactions.

```markdown
// Add your conclusion here.
```

## Database Alternatives

1. Weaviate: A cloud-native, modular, and real-time vector search engine that combines deep learning with a GraphQL API. It can be self-hosted or used as a managed service.
   - Repository: https://github.com/weaviate/weaviate
   - Documentation: https://www.semi.technology/documentation/weaviate/current/index.html

2. Pinecone: A managed vector database service that allows you to build, scale, and deploy applications using vector search.
   - Website: https://www.pinecone.io/
   - Documentation: https://docs.pinecone.io/reference/update



# Setup

You need to set up the following environment variables to share models
```
RUSTBERT_CACHE=/my/model/locaction
```