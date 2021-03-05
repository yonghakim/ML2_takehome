# Haar Cascade
미리 정해진 필터를 이용해 이미지를 스캔, 해당 필터의 패턴이 이미지에 존재하는지 여부를 검출<br>

<p align="center">
<img src="https://docs.opencv.org/3.4.1/haar_features.jpg">
</p>

자 봐라! 패턴 있지?  
<p align="center">
<img src="https://docs.opencv.org/3.4.1/haar.png">
</p>

3개만 알아가자
1. Integral image  
2. AdaBoost
3. Cascade

## 1. Integral image
패턴이 있는지 없는지 확인하는 방법이 무어냐? 간단하다. 그냥 Feature의 흰색부분에 해당하는
픽셀값의 합에서 어두운 부분의 픽셀값의 합을 빼주면 된다. 이게 일정 기준을 넘으면 해당 패턴이
있다고 판단. 근데 이걸 하려면 해당 필터 사이즈의 2차원 배열을 읽어야 함. 이걸 필터수만큼 반복하면
n^2 까진 아니지만(window size가 제한적이어서) 꽤나 시간이 많이 걸린다. 그래서 쓰이는 게 이
integral image.
######  
아래 그림을 보자

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/b/bd/Integral_image_application_example.svg" height=400>
</p>


1번 그림은 본래 이미지 배열. 2번은 본래 이미지의 적분영상 (integral image) 배열.
적분영상의 각각의 셀 값은 본래 이미지의 (0,0) 원점부터 구하고자 하는 해당 셀까지의 사각형에
포함되는 셀 값들의 합이다. 예로 그림 1의 빨간색 선((0,0)부터 (2,3)까지의 사각형)의 합은 101로
2번 그림의 (2,3)셀에 나타난다.  

다음으로 이 integral image를 이용해서 임의의 사각형에서의 픽셀 합을 구하려면 그림 2 아래의 방법
을 사용하면 된다. 수식은 위키(요기)를 참고.

## 2. AdaBoost
[link](Ensemble_Learning.md)

## 3. Cascade


## 참고 사이트
https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf  
https://www.codeproject.com/Articles/441226/Haar-feature-Object-Detection-in-Csharp  
https://en.wikipedia.org/wiki/Summed-area_table  

