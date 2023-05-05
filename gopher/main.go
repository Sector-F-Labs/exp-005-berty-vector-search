package main

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"

	"github.com/buckhx/gobert/model"
	"github.com/go-redis/redis/v8"
)

type BERTEmbedding struct {
	Values []float32 `json:"values"`
}

func readTextFiles(directory string) ([]string, error) {
	var texts []string
	err := filepath.Walk(directory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && filepath.Ext(path) == ".txt" {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			texts = append(texts, string(content))
		}
		return nil
	})
	return texts, err
}

func createBERTEmbeddings(texts []string) ([]BERTEmbedding, error) {
	// Implement BERT embeddings retrieval using an appropriate package
	// The embeddings should be of type []float32

	path := os.Getenv("RUSTBERT_CACHE")
	m, err := model.NewEmbeddings(path)
	if err != nil {
		panic(err)
	}
	res, err := m.PredictValues(texts...)
	if err != nil {
		panic(err)
	}

	var embeddings []BERTEmbedding
	for _, r := range res {
		embeddings = append(embeddings, BERTEmbedding{r})
	}

	return embeddings, nil
}

func storeEmbeddingsInRedis(redisClient *redis.Client, texts []string, embeddings []BERTEmbedding) error {
	ctx := context.Background()

	for i, text := range texts {
		hash := md5.Sum([]byte(text))
		key := fmt.Sprintf("embedding:%x", hash)

		embeddingJSON, err := json.Marshal(embeddings[i])
		if err != nil {
			return err
		}

		redisClient.HSet(ctx, key, "text", text)
		redisClient.HSet(ctx, key, "embedding", string(embeddingJSON))

		fmt.Printf("Stored embedding for document %d with key %s\n", i, key)
	}
	return nil
}

func cosineSimilarity(a, b []float32) float32 {
	var dotProduct, magnitudeA, magnitudeB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}
	return dotProduct / float32(math.Sqrt(float64(magnitudeA))*math.Sqrt(float64(magnitudeB)))
}

func retrieveDocumentsAndComputeSimilarity(redisClient *redis.Client, queryEmbedding BERTEmbedding) ([][2]interface{}, error) {
	ctx := context.Background()

	documentKeys, err := redisClient.Keys(ctx, "embedding:*").Result()
	if err != nil {
		return nil, err
	}

	var documentSimilarityScores [][2]interface{}
	for _, key := range documentKeys {
		var document BERTEmbedding
		embeddingJSON, err := redisClient.HGet(ctx, key, "embedding").Result()
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal([]byte(embeddingJSON), &document)
		if err != nil {
			return nil, err
		}

		similarity := cosineSimilarity(queryEmbedding.Values, document.Values)
		text, err := redisClient.HGet(ctx, key, "text").Result()
		if err != nil {
			return nil, err
		}

		documentSimilarityScores = append(documentSimilarityScores, [2]interface{}{text, similarity})
	}

	return documentSimilarityScores, nil
}

func main() {
	const url = "redis://127.0.0.1:6379"

	texts, err := readTextFiles("./texts")
	if err != nil {
		fmt.Printf("Error reading text files: %s\n", err)
		return
	}

	embeddings, err := createBERTEmbeddings(texts)
	if err != nil {
		fmt.Printf("Error creating embeddings: %s\n", err)
		return
	}

	redisClient := redis.NewClient(&redis.Options{
		Addr: url,
	})

	err = storeEmbeddingsInRedis(redisClient, texts, embeddings)
	if err != nil {
		fmt.Printf("Error storing embeddings in Redis: %s\n", err)
		return
	}

	query := "console.log"
	queryEmbedding, err := createBERTEmbeddings([]string{query})
	if err != nil {
		fmt.Printf("Error creating query embeddings: %s\n", err)
		return
	}

	documentSimilarityScores, err := retrieveDocumentsAndComputeSimilarity(redisClient, queryEmbedding[0])
	if err != nil {
		fmt.Printf("Error retrieving documents from Redis: %s\n", err)
		return
	}

	fmt.Printf("Similarity scores: %v\n", documentSimilarityScores)

}
