# Ensemble Learning
여러 개의 모델을 가지고 더 나은 하나의 모델을 만들어내는 것.

![](https://i2.wp.com/bdtechtalks.com/wp-content/uploads/2020/11/disparate-machine-learning-models.jpg?resize=696%2C321&ssl=1)
여러개의 모델들

![](https://i0.wp.com/bdtechtalks.com/wp-content/uploads/2020/11/ensemble-machine-learning-models.jpg?resize=696%2C392&ssl=1)
Ensemble 방법을 이용한 결과 개선  

간단한 예로 아래와 같은 기준으로 최종 모델을 정할 수 있따.
1. 다수 결정: 다득표한 후보를 리턴
2. 평균값: 각 모델에서 나온 값들의 평균값을 리턴
3. Weighted Averaging: 
   각 모델별로 가중치를 다르게 주어 Sum(가중치 * 각 모델의 결과값)을 리턴
   

## Common types of ensembles
### Bayes optimal classifier
TODO
### Bagging (Bootstrap aggregating)  
TODO

여러 모델이 병렬로 연결되어 더 나은 하나의 모델로 이어진다.  
Bagging, Pasting, Bootstrap  
* 용어  
  사람마다 정의가 조금씩 다르다.  
  Bagging: Bootstrap & Aggregating. Bootstrap 방법을 이용한 앙상블 메써드이시다.  
  Bootstrapping: 데이터를 샘플링할 때 뽑힌 아이를 Data Pool에 유지 (with replacement)  
  Pasting: 데이터를 샘플링할 때 뽑힌 아이를 Data Pool에서 제거 (w/o replacement)  

누군가는 Pasting을 Sampling method라고 하고 누군가는 이 샘플링 방법을 이용한 
앙상블 메써드를 지칭한다. 웹서치를 해보고 있으나 이거다싶은 레퍼런스를 찾기가 어려움.
제 개인적으론 Pasting과 Bootstrapping을 resampling 방법으로, Bagging은 
Bootstrap만을 이용한 앙상블 메써드로 이해할거임. 


### Boosting  
   !["afdf"](https://upload.wikimedia.org/wikipedia/commons/b/b5/Ensemble_Boosting.svg)  
   여러 개의 약한 모델(weak learner)로 이루어져 있고, 각 약한 모델별로 가중치를 적용해서
   하나의 최종 모델(strong learner)을 만들어낸다.
     
   * 빌드  
     여러 개의 약한 모델을 순차적으로 만든다.  
     앞선 모델에서 잘못 추론된 데이터에 가중치를 높게 주고 잘 
     추론된 경우 낮게 주어 새로운 데이터 셋을 만들고, 
     이 데이터를 이용해 앞선 모델을 보완하는 새 모델을 추가
     
   * 인퍼런스  
     앞서 만든 약한 모델들의 각각 결과를 구하고 이것들의 Weighted sum 
  
      
용어적인 측면에서 보면, Boosting은 Ensemble Model을 만들기 위한 __트레이닝__ 방법이다,
어떤 모델 종류를 의미하는 게 아니라. 다양한 부스팅 알고리즘이 있고 이들 사이의 주된 차이점은
트레이닝 데이터의 웨이팅 방법과 약한 Learning 에 쓰이는 모델 종류다. 
Wiki에는 weighting training data points와 hypotheses라고 나온다. 대표적으로 AdaBoost가 있고 
LPBoost, XGBoost 등이 있다.

1. AdaBoost(Adaptive Boost)  
    'adaptive하다' 이건 나중에 추가된 모델이 앞선 모델에서 잘못 분류된 데이터에 더 
       적합하도록 조정되는 문맥이다. 쉽고 빠른 이해를 원하면 요기, 수식을 보고 싶다면 요기로 가시길.
 


     

## 참고 사이트
https://en.wikipedia.org/wiki/Boosting_(machine_learning)  
https://bdtechtalks.com/2020/11/12/what-is-ensemble-learning/  
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
https://machinelearningmastery.com/why-use-ensemble-learning/#:~:text=An%20ensemble%20is%20a%20machine,on%20the%20same%20training%20data.
https://en.wikipedia.org/wiki/Ensemble_learning  
https://en.wikipedia.org/wiki/Boosting_(machine_learning)  
https://en.wikipedia.org/wiki/AdaBoost  
https://youtu.be/LsK-xG1cLYA  


