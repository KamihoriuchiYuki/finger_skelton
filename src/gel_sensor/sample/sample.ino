void setup() {
  // シリアル通信を開始
  Serial.begin(115200);
}

void loop() {
  // A0ピンのアナログ値を読み取る
  int sensorValue = analogRead(A0);
  
  // 読み取った値をシリアルモニタに出力
  Serial.println(sensorValue);
  
  // 1秒待つ
  delay(1000);
}
