import { Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as pickle from 'node-pickle';
import { SentimentLabels } from './enum/sentiment-labels.enum';
import * as path from 'path';

@Injectable()
export class AppService {
  private tokenizer: any;
  private model: tf.GraphModel;

  private modelPath = 'relative/path/to/your/model';

  constructor() {
    // this.loadModelV2();
    this.analyzeSentiment('I love python');
  }

  getHello(): string {
    return 'Hello!';
  }

  async loadModelV2() {
    const modelDir = path.join(__dirname, '..', 'src');
    console.log('Model ', modelDir);
    const model = await tf.loadLayersModel(`file://${modelDir}/out/model.json`);
    console.log('Model loaded successfully', model);
    return model;
  }

  async predictSentiment(text: string): Promise<string> {
    try {
      const tokens = this.tokenizer(text);
      const paddedTokens = tf.pad(tf.tensor2d([tokens]), [
        [0, 0],
        [0, 200 - tokens.length],
      ]);
      const prediction = this.model.predict(paddedTokens.reshape([1, 200]));
      // const combinedPrediction = tf.concat(prediction, 0); // Combine tensors in the array into a single tensor
      const predictedClassIndex = tf.argMax(prediction[0]).dataSync()[0];
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
    // TODO what to do
    this.tokenizer = pickle.loads(dataBuffer);
  }

  // private loadModel(): void {
  //   const modelPath = './model/model.json';
  //   this.model = tf.node.loadGraphModel(modelPath);
  // }
  async loadModel(modelPath: string): Promise<void> {
    try {
      this.model = await tf.loadGraphModel(`file://${modelPath}`);
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load model:', error);
      throw new Error('Failed to load model.');
    }
  }

  preprocessText(text: string) {
    const processedText = text.toLowerCase().replace(/[^a-z\s]/g, '');
    console.log('processedText ', processedText);
    return processedText;
  }

  tokenizeText(text: string) {
    const result = text.split(/\s+/);
    console.log('result ', result);
    return result;
  }

  analyzeSentiment(userInput: string) {
    const processedText = this.preprocessText(userInput);
    const tokenizedText = this.tokenizeText(processedText);
    const inputTensor = tf.tensor([[tokenizedText.length]]);
    console.log('inputTensor ', inputTensor);
  }
}
