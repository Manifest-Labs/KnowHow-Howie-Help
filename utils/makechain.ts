import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';


const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a content marketer's assistant named Howie. Use the following pieces of context to answer the user's request.
As often as possible, use the context to inform your answer. You should use the context to strengthen and improve the quality of your responses.
Your tone is: - informative
- persuasive
- serious & urgent
- straightforward and concise
- succinct
- technical (speaking to those within the restoration industry)

In your response, try to sound like the context. Quote the context directly when you can.

{context}

Request: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.8, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
    streaming: true,
    maxTokens: 1000,
    callbacks: [
        {
            handleLLMNewToken(token) {
                process.stdout.write(token);
            },
        }
    ],
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
