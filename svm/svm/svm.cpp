#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <vector>


// This is a SVM
// Basic on Li Hang 《统计学习方法》

// 浮点型向量
typedef std::vector<float> FLOATVECTOR;
// 整型向量
typedef std::vector<int> INTVECTOR;
// 浮点型矩阵
typedef std::vector<FLOATVECTOR> FLOATMAT;
// 整形矩阵
typedef std::vector<INTVECTOR> INTMAT;

// 从文件中读取数据及分类标签
// 输入:文件路径, 读取行数
void LoadDate(const char *filename,int line,
              FLOATMAT &datamat,INTVECTOR &labellist) {

  // 打开文件
  std::ifstream inFile(filename, std::ios::in);
  if (inFile.fail()) {
    puts("读取文件失败！");
  }
  // CSV文件中一行数据的容器
  std::string lineStr;
  // 从文件中读取一行数据到lineStr中
  int i = 0;
  while (getline(inFile, lineStr) && line--) {
    // str 用来装逗号隔开的数据
    std::string str;
    // 这一步参考网上的
    std::stringstream ss(lineStr);
    // 设置一个数据读取的标志位
    int start = 0;
    FLOATVECTOR datalists;
    // 分割lineStr这行数据逗号隔开的每一个分别存储
    while (getline(ss, str, ',')) {
      // 如果是这行的第一个数字（标签）
      if (!start) {
        // atoi <stdlib.h> 将字符串中的数据转换为int型
        int label = atoi(str.c_str());
        if (label == 0) {
          labellist.push_back(1);
          printf("第%d标签：1\n", ++i);
        } else {
          labellist.push_back(-1);
          printf("第%d标签：-1\n", ++i);
        }
        
        start = 1;
        continue;
      }
       // 转换到0~1之间的数
       float data = (float)atoi(str.c_str()) / 255;
       datalists.push_back(data);
    }
    datamat.push_back(datalists);
  }
  printf("读入数据行数：%d\n", i);
}

// SVM类
class SVM {

public:
  // SVM相关参数初始化
  // traindatalist:训练用数据集
  // trainlabellist:训练用标签集
  // sigma:高斯核中的分母sigma
  // C:惩罚系数
  // toler:松弛变量
  SVM(FLOATMAT &datalist, INTVECTOR &labellist, 
      float sigma, float C, float toler) {
    traindatalist = datalist;
    trainlabellist = labellist;
    if (!datalist.size()) {
      puts("data is NULL!!");
      return;
    }
    // 这一步一定要判空
    m = datalist.size();
    n = datalist[0].size();
    sigmas = sigma;
    Cs = C;
    tolers = toler;
    b = 0;
    
    CalcKernal();
    // 给所有的alpha付出值为0，长度为训练集数目
    alpha = FLOATVECTOR(m, 0);
    // 给所有的Ei也付初值为0，长度为训练集数目
    E = FLOATVECTOR(m, 0);
    // 支持向量数组
    //supportvecindex = FLOATVECTOR(m, 0);
  }

  // 核函数计算
  // 使用的是高斯核函数，详见“7.3.3 常用核函数” 式7.90
  // 计算结果存入kernelanswer矩阵中
  void CalcKernal() {
    // 初始化高斯核结果矩阵大小 = 训练集的长度的平方
    // 令有m个向量，每个向量里面m个元素
    // kernelanswer[i][j] = xi * xj;
    kernelanswer = FLOATMAT(m, FLOATVECTOR(m, 0));
    
    // 大循环遍历xi
    for (int i = 0; i < m; ++i) {
      // 向量xi
      FLOATVECTOR X = traindatalist[i];
      // 小循环遍历，只用计算一半即可
      for (int j = i; j < m; ++j) {
        FLOATVECTOR Z = traindatalist[j];
        // ||X-Z||^2
        float a = 0;
        for (int s = 0; s < n; ++s) {
          a += (X[s] - Z[s]) * (X[s] - Z[s]);
        }
        //计算出高斯核结果
        a = exp(-a / (2 * sigmas * sigmas));
        kernelanswer[i][j] = a;
        kernelanswer[j][i] = a;
      }
    }
    printf("K784784:%f\n", kernelanswer[999][999]);
  }
 
  // kkt条件满足判断
  // 查看第i个alpha是否满足kkt条件
  // return true,满足，false，不满足
  bool IsSatisfyKKT(int i) {
    // 算出g(x)
    float gxi = CalcGxi(i);
    int yi = trainlabellist[i];

    // 根据“7.4.2 变量的选择方法”中“第一个变量的选择”
    // 7.111 到 7.113
    if ((fabs(alpha[i]) < tolers) && (yi * gxi >= 1)){
      return true;
    } else if ((fabs(alpha[i] - Cs) < tolers) && (yi * gxi <= 1)) {
      return true;
    } else if ((alpha[i] > -tolers) && (alpha[i] < (Cs + tolers)) && \
               (fabs(yi * gxi - 1) < tolers)) {
      return true;
    } 
    return false;
  }

  // 计算i点到超平面距离
  // 按照7.104式子
  // return 距离
  float CalcGxi(int i) {
    /*
    因为g(xi)是一个求和式 + b的形式，普通做法应该是直接求出求和式中的每一项再相加即可
    但是发现，在“7.2.3 支持向量”开头第一句话有说到“对应于α > 0的样本点
    (xi, yi)的实例xi称为支持向量”。也就是说只有支持向量的α是大于0的，在求和式内的
    对应的αi*yi*K(xi, xj)不为0，非支持向量的αi*yi*K(xi, xj)必为0，也就不需要参与
    到计算中。也就是说，在g(xi)内部求和式的运算中，只需要计算α > 0的部分，其余部分可
    忽略。因为支持向量的数量是比较少的，这样可以再很大程度上节约时间
    */
    // 初始化gxi
    float gxi = 0;
    // 如果不为零就加上
    for (int j = 0; j < m; ++j) {
      if (alpha[j] != 0) {
        gxi += alpha[j] * trainlabellist[j] * kernelanswer[j][i];
      }
    }
    gxi += b;
    return gxi;
  }

  // 计算误差
  // 按照式子 7.105
  float CalcEi(int i) {
    return (CalcGxi(i) - trainlabellist[i]);
  }

  // 获得第二个alpha
  // 第一个alpha通过不符合KKT条件选取
  // 输入：E1，第一个变量的误差，i,第一个alpha
  // 返回第二个alpha的序号
  int GetAlphaJ(int E1, int i) {
    int alpha2_index = -1;
    // 第二个变量选取的标准是E1-E2
    float maxE1_E2 = -1;
    // E数组中的Ei最开始都是0所以只要遍历不为0的Ei即可
    for (int j = 0; j < m; ++j) {
      if (E[j] != 0) {
        E[j] = CalcEi(j);
        if (maxE1_E2 < fabs(E1 - E[j])) {
          maxE1_E2 = fabs(E1 - E[j]);
          alpha2_index = j;
        }
      }
    }
    // 若E数组里面没有非0的，就直接取出i后面这个
    if (maxE1_E2 == -1) {
      alpha2_index = (i + 2) % m;
    }
    return alpha2_index;
  }

  // SVM训练，SMO算法 
  // 输入迭代次数
  void Train(int count) {
    // 数据改变标志位
    int paramerter_changed = 1;

    // 迭代次数
    int iter = 0;
    // 如果已经达到迭代次数，或者已经没有参数改变-->认为已经达到收敛状态
    while (iter < count && paramerter_changed > 0) {
      // 打印当前迭代次数
      printf("iter: %d \n",iter);
      iter += 1;
      paramerter_changed = 0;

      // 一次迭代要遍历所有的alpha
      // 大循环找出第一个变量
      for (int k = 0; k < m; ++k) {
        // 如果不满足KKT条件，就选出本次alpha
        if (!IsSatisfyKKT(k)) {

          // 先计算E1,alpha2newnew = alpha2old + yi(E1 - E2)/(K11 + K22 - 2K12）
          float E1 = CalcEi(k);
          // 选取第二个alpha
          int j = GetAlphaJ(E1,k);
          float E2 = CalcEi(j);
          // 所以到这里 k 是第一个alpha的下标
          // j 是第二个alpha的下标

          // 计算alpha2new的L和H
          // 若y1 != y2
          float L = 0, H = 0;
          if (trainlabellist[k] != trainlabellist[j]) {
            L = fmax(0, alpha[j] - alpha[k]);
            H = fmin(Cs, Cs + alpha[j] - alpha[k]);
          } else {
            L = fmax(0, alpha[k] + alpha[j] - Cs);
            H = fmin(Cs, alpha[k] + alpha[j]);
          }
          //如果L和H相等，说明不能再优化，跳出当前循环，进入下一个变量
          if (L == H) {
            continue;
          }
          // 计算alpha2new_new
          float K11 = kernelanswer[k][k];
          float K12 = kernelanswer[k][j];
          float K21 = kernelanswer[j][k];
          float K22 = kernelanswer[j][j];

          float alpha2new_new = alpha[j] + trainlabellist[j] * (E1 - E2) \
                                / (K11 + K22 - 2 * K12);
          // 剪切alpha2new_new赋值给alpha2
          if (alpha2new_new < L) {
            alpha2new_new = L;
          } else if (alpha2new_new > H) {
            alpha2new_new = H;
          }
          // 计算alpha1new
          // 遵照式子 7.109
          float alpha1new = alpha[k] + trainlabellist[k] * trainlabellist[j] \
                           * (alpha[j] - alpha2new_new);

          // 计算b
          // 依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
          float b1 = -1 * E1 - trainlabellist[k] * K11 * (alpha1new - alpha[k])\
                     - trainlabellist[j] * K21 * (alpha2new_new - alpha[j]) + b;
          float b2 = -1 * E2 - trainlabellist[k] * K12 * (alpha1new - alpha[k])\
                     - trainlabellist[j] * K22 * (alpha2new_new - alpha[j]) + b;

          // 依据alpha1和alpha2的值范围确定新的b
          if (alpha1new > 0 && alpha1new < Cs) {
            b = b1;
          } else if (alpha2new_new > 0 && alpha2new_new < Cs) {
            b = b2;
          } else {
            b = (b1 + b2) / 2;
          }

          // 如果改动过小就不增加parameter_changed
          if (fabs(alpha2new_new - alpha[j]) >= 0.00001) {
            paramerter_changed += 1;
          }

          // 更新alpha，E
          alpha[k] = alpha1new;
          alpha[j] = alpha2new_new;
          E[k] = CalcEi(k);
          E[j] = CalcEi(j);
        }
      }
      printf("迭代次数:%d，改变alpha个数:%d\n", iter, paramerter_changed);
    }
    printf(" b = %d\n", b);
    // 保存所有的支持向量的索引
    for (int s = 0; s < m; ++s) {
      if (alpha[s] > 0) {
        supportvecindex.push_back(s);
      }
    }
  }

  // 做预测的时候需要单独的计算核函数
  float PredictCalcSingleKernel(int i, int j) {
     // 向量xi是训练集中的支持向量,保存支持向量的数组里面
     FLOATVECTOR X = traindatalist[i];
     // 测试向量
     FLOATVECTOR Z = testdatalist[j];
     // ||X-Z||^2
     float a = 0;
     for (int s = 0; s < n; ++s) {
        a += (X[s] - Z[s]) * (X[s] - Z[s]);
     }
     //计算出高斯核结果
     a = exp(-a / (2 * sigmas * sigmas));
     return a;
  }
    
  // 预测
  int predict(int i) {
    float K_p = 0,result = 0;
    // 遍历所有的支持向量，计算求和公式
    for (int k = 0; k < supportvecindex.size(); ++k) {
      // 先单独的将预测的核函数求出来
      // 传入支持向量的索引和测试向量的索引
      // alpha[k:n] * y[k:n] * K[k:n,i]
      K_p = PredictCalcSingleKernel(supportvecindex[k],i);
      // 逐步求和
      result = result + K_p * trainlabellist[supportvecindex[k]] * alpha[supportvecindex[k]];
    }
    result += b;

    // sign()
    if (result < 0) {
      return (-1);
    } else {
      return (1);
    }
  }

  // 测试训练好的模型
  void Test() {
    int error_count = 0;
    // 遍历测试集所有样本
    for (int i = 0; i < testdatalist.size(); ++i) {
      //printf("预测结果为： %d\n", predict(i));
      int r = predict(i);
      printf("预测结果为：%d\n", r);
      if (r != testlabellist[i]){
        printf("出错索引：%d ，预测标签为：%d ,原来标签为：%d\n", i, r, testlabellist[i]);
        error_count += 1;
      }
    }
    printf("出错的个数：%d", error_count);
  }

  FLOATVECTOR supportvecindex;  // 支持向量数组
  FLOATVECTOR alpha;            // 拉格朗日乘子数组
  FLOATVECTOR E;                // SMO算法运算过程中的E的数组
  FLOATMAT traindatalist;       // 训练数据集
  INTVECTOR trainlabellist;     // 标签集
  FLOATMAT testdatalist;        // 测试数据集
  INTVECTOR testlabellist;      // 测试标签集
  FLOATMAT kernelanswer;        // 核函数结果矩阵
  int m;                        // 数据集长度  
  int n;                        // 样本特征数量
  float b;                      // SVM中的偏置
private:
  float sigmas;                 // 高斯核中的分母
  float Cs;                     // 惩罚系数
  float tolers;                 // 松弛变量
};


void main() {
  // 数据集和标签集矩阵
  FLOATMAT datalist, testlist;
  INTVECTOR labellist, testlabellist;

  // 从数据集中导入数据和标签
  LoadDate("C:\\Users\\shuangjiang ou\\Desktop\\Mnist\\mnist_train.csv",1000, datalist,labellist);
  // 从数据集中导入测试数据和标签
  LoadDate("C:\\Users\\shuangjiang ou\\Desktop\\Mnist\\mnist_test.csv", 100, testlist, testlabellist);
  // LoadDate("C:\\Users\\shuangjiang ou\\Desktop\\Mnist\\mnist_train.csv", 200, testlist, testlabellist);
  SVM svm(datalist, labellist, 10, 200, 0.001);
  svm.testdatalist = testlist;
  svm.testlabellist = testlabellist;
  
  svm.Train(1000);
  svm.Test();
  int i;
  scanf("%d", &i);

  //datalist = FLOATMAT(4, FLOATVECTOR(2));
  //labellist = INTVECTOR(4);
  //for (int i = 0; i < 4; ++i) {
  //  printf("输入第%d个向量：\n", i);
  //  float j = 0;
  //  scanf("%f", &j);
  //  datalist[i][0] = j;
  //  scanf("%f", &j);
  //  datalist[i][1] = j;
  //  puts("输入此向量的特征值");
  //  scanf("%f", &j);
  //  labellist[i] = j;
  //}
  //// 初始化一个SVM
  //SVM svm(datalist, labellist, 10, 200, 0.001);
  //svm.Train(1000);
  //for (int i = 0; i < svm.m; ++i) {
  //  for (int j = 0; j < svm.m; ++j) {
  //    printf("%f\t", svm.kernelanswer[i][j]);
  //  }
  //  printf("\n");
  //}
  //testlist = FLOATMAT(4, FLOATVECTOR(2));
  //for (int i = 0; i < 4; ++i) {
  //  printf("输入第%d个向量：\n", i);
  //  float j = 0;
  //  scanf("%f", &j);
  //  testlist[i][0] = j;
  //  scanf("%f", &j);
  //  testlist[i][1] = j;
  //}
  //svm.testdatalist = testlist;
  //svm.Test();
  //int j;
  //scanf("%d", &j);
}
