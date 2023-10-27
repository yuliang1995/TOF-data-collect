# TOF-data-collect
TOF是一種深度相機，他所呈現的影像為IR與Depth，為了驗證相機的FOV、距離、亮度。
因此撰寫一個程式來驗證並儲存內容。

主要記錄影像中的九點位置，如下圖。  

![image](https://github.com/yuliang1995/TOF-data-collect/blob/main/image/Depth0.jpg?raw=true)
![image](https://github.com/yuliang1995/TOF-data-collect/blob/main/image/IR0.jpg?raw=true)

  
FOV的計算是參考Leetcode.221的方法，去計算出白色部分有多少並透過畢氏定理運算出FOV
