import "dotenv/config";
import readline from 'readline';
import OpenAI from "openai";
import { GoogleGenAI } from "@google/genai";

const openaiClient = new OpenAI();

const geminiClient = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

// with zod validation there were some text.format.name error cause new text.format uses new properties so we created a custom json schema instead of using zod
const EvaluationJsonSchema = {
    type: "object",
    properties: {
        winner: { type: "string", description: "here we will get the winner output model" },
        reasoning: { type: "string", description: "here we will list the reason based on which it was made a winner" },
        winning_text: { type: "string", description: "here we will display winning response" }
    },
    required: ["winner", "reasoning", "winning_text"],
    additionalProperties: false
};

const chatHistory = [];

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function processPrompt(userPrompt) {
    console.log("\nParallel processing started");

    try {
        const [output_OpenAI, output_Gemini] = await Promise.all([
            // calling open ai sdk
            openaiClient.responses.create({
                model: "gpt-4o-mini",
                input: userPrompt,
                max_output_tokens: 150
            }).then((res) => {
                return res.output_text;
            }),

            // Calling gemini sdk
            geminiClient.models.generateContent({
                model: "gemini-2.5-flash",
                contents: userPrompt,
                config: {
                    maxOutputTokens: 150
                }
            }).then((res) => {
                return res.text;
            }),
        ]);

        console.log("Received all responses successfully");

        // Evaluation prompt
        const evaluationPrompt = `
            You are an expert AI arbiter evaluating competitive text models.
            The user's current request is: "${userPrompt}"

            There separate engines returned these outputs:
            [RESPONSE OPTION GPT-4o-Mini]:
            ${output_OpenAI}

            [RESPONSE OPTION Gemini Flash]:
            ${output_Gemini}

            Task: Choose the single best response based on thoroughness, correctness, and natural tone.
        `;

        // Setting up chat history alignment for the Responses API
        const judgeMessage = [
            {
                role: "system",
                content: "You are an objective judge assessing text quality."
            },
            ...chatHistory,
            {
                role: "user",
                content: evaluationPrompt
            }
        ];

        // API call for final best output selection, uses new text format rules
        const response = await openaiClient.responses.create({
            model: "gpt-4o-mini",
            input: judgeMessage,
            text: {
                format: {
                    type: "json_schema",
                    name: "evaluation",
                    strict: true,
                    schema: EvaluationJsonSchema
                }
            },
            temperature: 0.2,
            max_output_tokens: 300
        });

        const evaluationResult = JSON.parse(response.output_text);

        // Printing individual details to console for tracking comparison
        console.log("\nOutput given by LLMs");
        console.log(`[A] GPT-4o-Mini Preview:\n${output_OpenAI}\n`);
        console.log(`[B] Gemini Flash Preview:\n${output_Gemini}\n`);

        console.log('================ 🏆 WINNING SELECTION ================\n');
        console.log(`👑 Selected Winner: ${evaluationResult.winner}\n`);
        console.log(`🧠 Judge Reasoning: ${evaluationResult.reasoning}\n`);
        console.log(`📄 Content:\n${evaluationResult.winning_text}\n`);
        console.log('======================================================= \n');

        chatHistory.push({ role: "user", content: userPrompt });
        chatHistory.push({ role: "assistant", content: evaluationResult.winning_text });

    } catch (error) {
        console.log("\nPipeline execution failed:", error.message);
    }
}

function startCLI() {
    rl.question(`Ask your question (or type "exit" to quit): `, async (input) => {
        if (input.toLowerCase() === "exit") {
            rl.close();
            return;
        }

        if (!input.trim()) {
            startCLI();
            return;
        }

        await processPrompt(input);
        startCLI();
    });
}

console.log('=========Structured Multi-LLM Engine Started============');
startCLI();