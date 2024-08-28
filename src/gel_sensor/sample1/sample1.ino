void setup() {
  // シリアル通信を初期化
  Serial.begin(9600);
}

void loop() {
  // A0のアナログ入力を測定
  long sensorValue = analogRead(A0); //
  
  //sample
  // long sensorValue[5];
  // for(int i 05){
  //   sens[i]=val[i]
  // }
  // char val_enc[10];
  // for(int ){
  //   val_encoded[i*2]   = (sensorValue[i] & 0xFF);
  //   val_encoded[i*2+1] = (sensorValue[i] >> 8);
  // }
  // Serial.write(val_encoded,length(val_enc));  

  // sensorValue = 256;
  char val_encoded[2];
  val_encoded[0] = (sensorValue & 0xFF);
  val_encoded[1] = (sensorValue >> 8);
  // Serial.print(sensorValue);
  // 測定値をシリアルポート経由で送信
  Serial.write(val_encoded,2);    // 上位バイトを送信
  //Serial.write(sensorValue & 0xFF);  // 下位バイトを送信
  
  // 少し待つ（例：1秒間隔で測定）
  delay(10);
}
