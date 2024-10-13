using OpenAI;
using OpenAI.Chat;
using System.ClientModel;
using System.Text.Json;

Console.WriteLine("--------------------");
Console.WriteLine("Hello, Phi-3-Vision!");
Console.WriteLine("--------------------");

await Run();

static async Task Run()
{
    var prompt = "Please extract the items, price, tax, and totals including payment type from the receipt provided.";
    var imageFilePath = Path.Combine("Assets", "receipt_01.jpeg");
    var imageBytes = await File.ReadAllBytesAsync(imageFilePath);
    var base64String = Convert.ToBase64String(imageBytes);
    var base64ImageUrl = $"data:image/jpeg;base64,{base64String}";
    var chatHistory = new List<ChatMessage>();

    chatHistory.Add(new UserChatMessage(
        ChatMessageContentPart.CreateTextPart(prompt),
        ChatMessageContentPart.CreateImagePart(new Uri(base64ImageUrl))));

    var completion = await GetImageDetailsWithMessages(chatHistory);

    Console.WriteLine($"[ASSISTANT]: {completion.Content[0].Text}");
    Console.WriteLine();

    // Append the assistant's response to the chat history
    chatHistory.Add(new AssistantChatMessage(
    ChatMessageContentPart.CreateTextPart(completion.Content[0].Text)));

    // Second user message (follow-up question)
    chatHistory.Add(new UserChatMessage(
        ChatMessageContentPart.CreateTextPart("What type of credit card was used to pay for the items?")));

    completion = await GetImageDetailsWithMessages(chatHistory);
    
    Console.WriteLine($"[ASSISTANT]: {completion.Content[0].Text}");
}

static async Task<ChatCompletion> GetImageDetailsWithMessages(List<ChatMessage> messages)
{
    var apiKey = "API_KEY";
    var endpoint = "http://localhost:81/";
    var apiKeyCerdential = new ApiKeyCredential(apiKey);

    OpenAIClient openAIClient = new(apiKeyCerdential, new OpenAIClientOptions() { Endpoint = new Uri(endpoint) });
    ChatClient chatClient = openAIClient.GetChatClient("Phi-3.5-mini-instruct");

    return await chatClient.CompleteChatAsync(messages);
}

static async Task<ChatCompletion> GetImageDetails(string imagePath, string prompt)
{
    var apiKey = "API_KEY";
    var endpoint = "http://localhost:81/";
    var apiKeyCerdential = new ApiKeyCredential(apiKey);

    OpenAIClient openAIClient = new(apiKeyCerdential, new OpenAIClientOptions() { Endpoint = new Uri(endpoint) });
    ChatClient chatClient = openAIClient.GetChatClient("Phi-3.5-mini-instruct");

    var imageStream = File.OpenRead(imagePath);
    BinaryData imageBytes = BinaryData.FromStream(imageStream);

    List<ChatMessage> messages =
    [
        new UserChatMessage(
                ChatMessageContentPart.CreateTextPart(prompt),
                ChatMessageContentPart.CreateImagePart(imageBytes, "image/png")),
        ];

    return await chatClient.CompleteChatAsync(messages);
}