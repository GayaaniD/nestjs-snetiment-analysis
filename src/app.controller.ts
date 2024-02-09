import { Controller, Get, Post, Body } from '@nestjs/common';
import { AppService } from './app.service';
import { InputText } from './dto/input-text.dto';


@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get()
  getHello(): string {
    return this.appService.getHello();
  }

  @Post('/predict')
  async predictSentiment(@Body() input: InputText) {
    return this.appService.predictSentiment(input.text);
  }
}
