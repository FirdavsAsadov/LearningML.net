using System;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)]
    public string Text;

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment;
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    [ColumnName("Probability")]
    public float Probability { get; set; }

    [ColumnName("Score")]
    public float Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        var data = new[]
        {
            new SentimentData {Text = "This is a great product", Sentiment = true},
            new SentimentData {Text = "I hate this thing", Sentiment = false},
            // Ko'proq ma'lumotlar qo'shing
        };

        var dataView = mlContext.Data.LoadFromEnumerable(data);

        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

        var model = pipeline.Fit(dataView);

        // Foydalanuvchi kiritgan matn uchun sentiment tahlili
        while (true)
        {
            Console.WriteLine("Enter a text to analyze sentiment (or 'exit' to quit):");
            string inputText = Console.ReadLine();
            if (inputText.ToLower() == "exit")
            {
                break;
            }
            var predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            var sampleStatement = new SentimentData { Text = inputText };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine($"Text: '{inputText}' is {(resultPrediction.Prediction ? "Positive" : "Negative")}");
            Console.WriteLine(RespondBasedOnSentiment(resultPrediction.Prediction));
        }
    }

    static string RespondBasedOnSentiment(bool sentiment)
    {
        if (sentiment)
        {
            return "Thank you for your positive comment!";
        }
        else
        {
            return "Sorry to hear that. We strive to improve!";
        }
    }
}
