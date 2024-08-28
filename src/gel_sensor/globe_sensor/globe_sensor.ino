void setup() {
  // シリアル通信を初期化
  Serial.begin(19200);
}

void loop() {
  long sensorValues[5]; // A0からA4までのセンサー値を格納する配列
  char val_encoded[10]; // 各センサー値を2バイトにエンコードしたものを格納する配列

  // A0からA4までのアナログ入力を測定
  for (int i = 0; i < 5; i++) {
    sensorValues[i] = analogRead(A0 + i);
    
    // 測定したセンサー値を2バイトにエンコード
    val_encoded[i * 2]     = (sensorValues[i] & 0xFF);      // 下位バイト
    val_encoded[i * 2 + 1] = ((sensorValues[i] >> 8) & 0xFF);        // 上位バイト
  }

  // エンコードされたセンサー値をシリアルポート経由で送信
  Serial.write(val_encoded, sizeof(val_encoded));

  // 少し待つ（例：1秒間隔で測定）
  delay(100);
}

