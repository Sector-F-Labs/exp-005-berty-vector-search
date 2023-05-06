import fs from 'fs';
import path from 'path';
import axios from 'axios';

type EmbeddingResponse = {
  data: {
    embedding: number[]
  }[]
};

type EmbeddingValue = {
  values: number[];
  text: string;
};

type ScoredDocument = {
  text: string;
  similarity: number;
};

const readTextFiles = (directory: string): string[] => {
  const texts: string[] = [];
  const files = fs.readdirSync(directory);

  files.forEach((file) => {
    const filePath = path.join(directory, file);
    const isTextFile = path.extname(file) === '.txt';

    if (isTextFile) {
      const content = fs.readFileSync(filePath, 'utf-8');
      texts.push(content);
    }
  });

  return texts;
};

async function getEmbedding(apiKey: string, inputText: string, model: string): Promise<EmbeddingResponse> {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/embeddings',
      {
        input: inputText,
        model: model,
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching embeddings:', error);
    throw error;
  }
}


const createEmbeddings = async (texts: string[]): Promise<EmbeddingValue[]> => {
  const apiKey = process.env['OPENAI_API_KEY'] || '';
  if (!apiKey) throw new Error('OPENAI_API_KEY environment variable is not set');

  const embeddings: EmbeddingValue[] = await Promise.all(texts.map(async (text) => { 
    const er =  await getEmbedding(apiKey, text, 'text-embedding-ada-002');
    return {
      values: er.data[0].embedding,
      text: text
    }
  }));

  return embeddings
};

const cosineSimilarity = (a: number[], b: number[]): number => {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

const computeSimilarity = async (
  embeddings: EmbeddingValue[],
  query: number[],
): Promise<ScoredDocument[]> => {
  const scores = []

  for (const existing of embeddings) {
    console.log(`query: ${query} existing: ${existing.values}`);
    const similarity = cosineSimilarity(query, existing.values);
    const scored: ScoredDocument = {
      text: existing.text,
      similarity,
    };
    scores.push(scored);
  }

  return scores;
};


const main = async () => {
    const directory = path.join(__dirname, 'texts');
    const texts = readTextFiles(directory);

    let embeddings: EmbeddingValue[] = [];
    try {
        embeddings = await createEmbeddings(texts);
    } catch (e: unknown) {
        console.log('Failed to create embeddings',e); 
    }

    const query = 'some rust code'
    
    const [queryEmbedding]  = await createEmbeddings([query]);
    

    const similarityScores = await computeSimilarity(embeddings, queryEmbedding.values);
    console.log(similarityScores);

    //choose option with highest similarity score

    const sorted = similarityScores.sort((a, b) => b.similarity - a.similarity);
    const winningText = sorted[0].text;
    console.log(`Found ${query} in  text: ${winningText}`);

}
main();