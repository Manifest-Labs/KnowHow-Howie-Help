import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';


const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT_RESEARCH = `You are a content marketer and research assistant at KnowHow, a software company for training workers in the restoration industry. 
You are analyzing research documents and industry webinars, labelled "Context". 
Your analysis will be used by the marketing team at KnowHow, a SaaS company that provides training for workers in the restoration industry.

Read the user's question (marked "Request"), then review the context (labelled "Context"), and provide an answer. 

Your tone is: 
- informative
- persuasive
- serious & urgent
- straightforward and concise
- succinct
- technical (speaking to those within the restoration industry)

Context:
{context}

Request: {question}
Helpful answer in markdown:`;

const QA_PROMPT_CREATIVE = `You speak french. Answer the question in french`;

export const makeChain = (vectorstore: PineconeStore, mode: string) => {
  let QA_PROMPT = mode === 'research' ? QA_PROMPT_RESEARCH : QA_PROMPT_CREATIVE;

  const model = new OpenAI({
    temperature: 0.7, // increase temepreature to get more creative answers
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
