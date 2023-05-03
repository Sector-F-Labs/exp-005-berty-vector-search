package main

import (
	"context"
	"fmt"

	client "github.com/milvus-io/milvus-sdk-go/v2"
)

func main() {
	URL := "http://localhost:19121"

	client, err := client.NewGrpcClient(context.Background(), URL)
	if err != nil {
		// handle error
		fmt.Println(err)
	}

	fmt.Println("Lets go")

	defer client.Close()

	client.HasCollection(context.Background(), "YOUR_COLLECTION_NAME")
}
