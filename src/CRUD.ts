import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import { QdrantClient } from '@qdrant/js-client-rest';

// Load environment variables
dotenv.config();

// Environment variable for OpenAI API key
const openaiApiKey = process.env.OPENAI_API_KEY!;
const openai = new OpenAI({
  apiKey: openaiApiKey, // This is the default and can be omitted
});

// Initialize Qdrant client
const client = new QdrantClient({ host: "localhost", port: 6333 });

// Collection name in Qdrant
const collectionName = 'example_collection';

// Function to get OpenAI embeddings for a given text
async function getOpenAIEmbedding(text: string) {
  const embedding = await openai.embeddings.create({
    model: 'text-embedding-ada-002', // You can choose other models if needed
    input: text,  // Text you want to embed
  });
  return embedding.data[0].embedding; // Returns the embedding vector
}

// Function to create or upsert points in Qdrant
async function createOrUpdatePoint(id: string | number, text: string) {
  const embedding = await getOpenAIEmbedding(text);
  const point = {
    id,
    vector: embedding,
    payload: { text },  // You can add more metadata to the payload if necessary
  };

  await client.upsert(collectionName, {
    wait: true,
    points: [point],
  });

  console.log(`Point with ID ${id} has been upserted!`);
}

// Function to read (retrieve) points from Qdrant
async function readPoint(id: string | number) {
  const points = await client.retrieve(collectionName, {
    ids: [id],
  });

  if (points.length === 0) {
    console.log(`No point found with ID ${id}. It may have been deleted.`);
  } else {
    console.log(`Retrieved points:`, points);
  }

  return points;
}

// Function to delete points from Qdrant
async function deletePoints(ids: (string | number)[]) {
  const response = await client.delete(collectionName, {
    wait: true,
    points: ids, // Array of point IDs to delete
  });
  console.log('Delete operation result:', response);
}

// Function to search for facts in Qdrant based on a query
async function searchForFacts(query: string) {
  const embedding = await getOpenAIEmbedding(query);  // Get the embedding for the query
  const searchResults = await client.search(collectionName, {
    vector: embedding,
    limit: 3,  // Limit the number of results for brevity
  });

  return searchResults;
}

// Function to generate a response from OpenAI based on a query and facts from Qdrant
async function generateResponse(query: string) {
  // Search for relevant facts in the database
  const searchResults = await searchForFacts(query);

  if (searchResults.length === 0) {
    console.log("No relevant facts found in the database.");
    return "I could not find any relevant information.";
  }

  // Combine the results (facts) to form the context for OpenAI's response
  const facts = searchResults.map((result) => result.payload?.text).join("\n");

  // Generate a response using OpenAI and the retrieved facts
  const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [
      {
        role: 'system',
        content: 'You are an assistant that answers questions based on facts from a knowledge base.',
      },
      {
        role: 'user',
        content: `Here are some relevant facts:\n${facts}\n\nPlease respond to the following question: ${query}`,
      },
    ],
  });

  return response.choices[0].message.content; // Return the response from OpenAI
}

// Main function to demonstrate the functionality
async function main() {
  const fact1 = "The Earth orbits the Sun.";
  const fact2 = "The human body contains 206 bones.";
  const fact3 = "The Eiffel Tower is located in Paris, France.";

  // Insert (Upsert) facts into the database
  await createOrUpdatePoint(1, fact1);
  await createOrUpdatePoint(2, fact2);
  await createOrUpdatePoint(3, fact3);

  // Generate a response based on a user query
  const query = "How many bones are in the human body?";
  const response = await generateResponse(query);
  console.log("Generated response:", response);

  // Read the points used to generate the response
  await readPoint(1);
  await readPoint(2);

  // Delete the points used in the response
  await deletePoints([1, 2]);

  // Try reading the deleted points
  await readPoint(1);
  await readPoint(2);

  // You can also try querying about the Earth or the Eiffel Tower
  const query2 = "Where is the Eiffel Tower located?";
  const response2 = await generateResponse(query2);
  console.log("Generated response:", response2);

  // Read and delete the point used for the second query
  await readPoint(3);
  await deletePoints([3]);
}

// Execute the main function
main().catch(console.error);
