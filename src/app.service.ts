import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as pickle from 'node-pickle';
import { SentimentLabels } from './enum/sentiment-labels.enum';

@Injectable()
export class AppService {
  private tokenizer: any;
  private model: tf.GraphModel;

  constructor() {
    this.loadTokenizer();
    this.loadModel();
  }

  getHello(): string {
    return 'Hello World!';
  }

  async predictSentiment(text: string): Promise<string> {
    try {
      const tokens = this.tokenizer(text);
      const paddedTokens = tf.pad(tf.tensor2d(tokens), [[0, 0], [0, 200 - tokens.length]]);
      const prediction = this.model.predict(paddedTokens.reshape([1, 200]));
      const predictedClassIndex = tf.argMax(prediction, 1).dataSync()[0];
      const predictedSentiment = SentimentLabels[predictedClassIndex];
      console.log(`Predicted Sentiment: ${predictedSentiment}`);
      return predictedSentiment;
    } catch (error) {
      console.error(`Failed to predict sentiment. Error: ${error}`);
      throw new Error('Failed to predict sentiment.');
    }
  }

  private loadTokenizer(): void {
    const tokenizerPath = 'vectorizer.pkl';
    const dataBuffer = fs.readFileSync(tokenizerPath);
    this.tokenizer = pickle.loads(dataBuffer);
  }

  private loadModel(): void {
    const modelPath = './model/model.json';
    this.model = tf.node.loadGraphModel(modelPath);
  }
}
