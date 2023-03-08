from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
input_file = open('review_body_pacifier.txt',encoding='utf-8')
output_file = open("pinglun_sentiment.txt", mode="w",encoding='utf-8')
csv_writer = csv.writer(output_file)
sentences = input_file.readlines()
analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
	vs = analyzer.polarity_scores(sentence)
	csv_writer.writerow([sentence,vs['compound']])
	output_file.write(str(sentence)+str(vs['compound'])+'n')
input_file.close()
output_file.close()