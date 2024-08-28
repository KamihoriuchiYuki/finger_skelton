#include <Arduino.h>


// A0ピンからアナログ入力を読み取り、シリアルモニタに出力するスケッチ

void setup() {
  // シリアル通信を開始し、ボーレートを9600bpsに設定
  Serial.begin(9600);

  // A0ピンを入力モードに設定（デフォルトでアナログ入力ピンは入力モード）
  pinMode(A0, INPUT);
}

void loop() {
  // A0ピンからアナログ入力を読み取る
  int sensorValue = analogRead(A0);

  // 読み取った値をシリアルモニタに出力
  Serial.print("A0 Input: ");
  Serial.println(sensorValue);

  // 500ミリ秒待つ
  delay(500);
}
