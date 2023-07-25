import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Load document
const loader = new TextLoader("data.txt");
const docs = await loader.load();

// Split the Document into chunks for embedding and vector storage
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
});

const splitDocs = await textSplitter.splitDocuments(docs);

// Embed and store the splits in an in-memory vector database
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

// Query the retrieved documents for an answer using LLM
const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(),
    {
        returnSourceDocuments: true
    });

const response = await chain.call({
  query: "What is task decomposition?"
});

// output
console.log(response.sourceDocuments[0].pageContent);
console.log(response.sourceDocuments[0].metadata);
