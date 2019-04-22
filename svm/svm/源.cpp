
/*********************************************************

**CopyRight by Weidi Xu, S.C.U.T in Guangdong, Guangzhou**

**********************************************************/



#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
using std::sort;
using std::fabs;

const int MAX_DIMENSION = 2;
const int MAX_SAMPLES = 3;
double x[MAX_SAMPLES][MAX_DIMENSION];
double y[MAX_SAMPLES];
double alpha[MAX_SAMPLES];
double w[MAX_DIMENSION];
double b;
double c;
double eps = 1e-6;

struct _E {
  double val;
  int index;
}E[MAX_SAMPLES];

bool cmp(const _E & a, const _E & b)
{
  return a.val < b.val;
}

// 向量维数
int num_dimension;

// 向量个数
int num_samples;

double max(double a, double b)
{
  return a > b ? a : b;
}

double min(double a, double b)
{
  return a > b ? b : a;
}



double kernal(double x1[], double x2[], double dimension)
{
  double ans = 0;
  for (int i = 0; i < dimension; i++)
  {
    ans += x1[i] * x2[i];
  }
  return ans;
}

double target_function()
{
  double ans = 0;
  for (int i = 0; i < num_samples; i++)
  {
    for (int j = 0; j < num_samples; j++)
    {
      ans += alpha[i] * alpha[j] * y[i] * y[j] * kernal(x[i], x[j], num_dimension);
    }
  }

  for (int i = 0; i < num_samples; i++)
  {
    ans -= alpha[i];
  }
  return ans;
}

double g(double _x[], int dimension)
{
  double ans = b;
  for (int i = 0; i < num_samples; i++)
  {
    ans += alpha[i] * y[i] * kernal(x[i], _x, dimension);
  }
  return ans;

}



bool satisfy_constrains(int i, int dimension)
{
  if (alpha[i] == 0)
  {
    if (y[i] * g(x[i], dimension) >= 1)
      return true;
    else
      return false;
  }
  else if (alpha[i] > 0 && alpha[i] < c)
  {
    if (y[i] * g(x[i], dimension) == 1)
      return true;
    else
      return false;
  }
  else
  {
    if (y[i] * g(x[i], dimension) <= 1)
      return true;
    else
      return false;
  }
}

double calE(int i, int dimension)
{
  return g(x[i], dimension) - y[i];
}



void calW()

{

  for (int i = 0; i < num_dimension; i++)

  {

    w[i] = 0;

    for (int j = 0; j < num_samples; j++)

    {

      w[i] += alpha[j] * y[j] * x[j][i];

    }

  }

  return;

}

void calB()

{

  double ans = y[0];

  for (int i = 0; i < num_samples; i++)

  {

    ans -= y[i] * alpha[i] * kernal(x[i], x[0], num_dimension);

  }

  b = ans;

  return;

}

void recalB(int alpha1index, int alpha2index, int dimension, double alpha1old, double alpha2old)

{

  double alpha1new = alpha[alpha1index];

  double alpha2new = alpha[alpha2index];



  alpha[alpha1index] = alpha1old;

  alpha[alpha2index] = alpha2old;



  double e1 = calE(alpha1index, num_dimension);

  double e2 = calE(alpha2index, num_dimension);



  alpha[alpha1index] = alpha1new;

  alpha[alpha2index] = alpha2new;



  double b1new = -e1 - y[alpha1index] * kernal(x[alpha1index], x[alpha1index], dimension)*(alpha1new - alpha1old);

  b1new -= y[alpha2index] * kernal(x[alpha2index], x[alpha1index], dimension)*(alpha2new - alpha2old) + b;



  double b2new = -e2 - y[alpha1index] * kernal(x[alpha1index], x[alpha2index], dimension)*(alpha1new - alpha1old);

  b1new -= y[alpha2index] * kernal(x[alpha2index], x[alpha2index], dimension)*(alpha2new - alpha2old) + b;



  b = (b1new + b2new) / 2;

}

bool optimizehelp(int alpha1index, int alpha2index)
{
  double alpha1new = alpha[alpha1index];
  double alpha2new = alpha[alpha2index];
  double alpha1old = alpha[alpha1index];
  double alpha2old = alpha[alpha2index];
  double H, L;
  if (fabs(y[alpha1index] - y[alpha2index]) > eps)
  {
    L = max(0, alpha2old - alpha1old);
    H = min(c, c + alpha2old - alpha1old);
  }
  else
  {
    L = max(0, alpha2old + alpha1old - c);
    H = min(c, alpha2old + alpha1old);
  }
  //cal new
  double lena = kernal(x[alpha1index], x[alpha1index], num_dimension) + kernal(x[alpha2index],
           x[alpha2index], num_dimension) - 2 * kernal(x[alpha1index], x[alpha2index], num_dimension);
  alpha2new = alpha2old + y[alpha2index] * (calE(alpha1index, num_dimension)
              - calE(alpha2index, num_dimension)) / lena;
  if (alpha2new > H)
  {
    alpha2new = H;
  }
  else if (alpha2new < L)
  {
    alpha2new = L;
  }
  alpha1new = alpha1old + y[alpha1index] * y[alpha2index] * (alpha2old - alpha2new);
  double energyold = target_function();
  alpha[alpha1index] = alpha1new;
  alpha[alpha2index] = alpha2new;
  double gap = 0.001;
  recalB(alpha1index, alpha2index, num_dimension, alpha1old, alpha2old);
  return true;
}

bool optimize()
{
  int alpha1index = -1;
  int alpha2index = -1;
  double alpha2new = 0;
  double alpha1new = 0;
  //cal E[]

  // 计算E
  for (int i = 0; i < num_samples; i++)
  {
    E[i].val = calE(i, num_dimension);
    E[i].index = i;
  }
  //traverse the alpha1index with 0 < && < c
  for (int i = 0; i < num_samples; i++)
  {
    alpha1new = alpha[i];
    if (alpha1new > 0 && alpha1new < c)
    {
      if (satisfy_constrains(i, num_dimension))
        continue;
      sort(E, E + num_samples, cmp);
      //simply find the maximum or minimun;
      if (alpha1new > 0)
      {
        if (E[0].index == i)
        {
          ;
        }
        else

        {

          alpha1index = i;

          alpha2index = E[0].index;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }

      else

      {

        if (E[num_samples - 1].index == i)

        {

          ;

        }

        else

        {

          alpha1index = i;

          alpha2index = E[num_samples - 1].index;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }





      //find the alpha2 > 0 && < c

      for (int j = 0; j < num_samples; j++)

      {

        alpha2new = alpha[j];



        if (alpha2new > 0 && alpha2new < c)

        {

          alpha1index = i;

          alpha2index = j;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }



      //find other alpha2

      for (int j = 0; j < num_samples; j++)

      {

        alpha2new = alpha[j];



        if (!(alpha2new > 0 && alpha2new < c))

        {

          alpha1index = i;

          alpha2index = j;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }

    }

  }



  //find all alpha1

  for (int i = 0; i < num_samples; i++)

  {

    alpha1new = alpha[i];



    if (!(alpha1new > 0 && alpha1new < c))

    {

      if (satisfy_constrains(i, num_dimension))

        continue;



      sort(E, E + num_samples, cmp);



      //simply find the maximum or minimun;

      if (alpha1new > 0)

      {

        if (E[0].index == i)

        {

          ;

        }

        else

        {

          alpha1index = i;

          alpha2index = E[0].index;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }

      else

      {

        if (E[num_samples - 1].index == i)

        {

          ;

        }

        else

        {

          alpha1index = i;

          alpha2index = E[num_samples - 1].index;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }





      //find the alpha2 > 0 && < c

      for (int j = 0; j < num_samples; j++)

      {

        alpha2new = alpha[j];



        if (alpha2new > 0 && alpha2new < c)

        {

          alpha1index = i;

          alpha2index = j;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }



      //find other alpha2

      for (int j = 0; j < num_samples; j++)

      {

        alpha2new = alpha[j];



        if (!(alpha2new > 0 && alpha2new < c))

        {

          alpha1index = i;

          alpha2index = j;

          if (optimizehelp(alpha1index, alpha2index))

          {

            return true;

          }

        }

      }

    }

  }



  //for(int i = 0 ; i < num_samples; i++)

  //{

  //    alpha1new = alpha[i];



  //    for(int j = 0 ; j < num_samples; j++)

  //    {

  //        if(1)

  //        {

  //            alpha1index = i;

  //            alpha2index = j;

  //            if(optimizehelp(alpha1index , alpha2index))

  //            {

  //                return true;

  //            }

  //        }

  //    }

  //}

  return false;

}



bool check()

{

  double sum = 0;

  for (int i = 0; i < num_samples; i++)

  {

    sum += alpha[i] * y[i];

    if (!(0 <= alpha[i] && alpha[i] <= c))

    {

      printf("alpha[%d]: %lf wrong\n", i, alpha[i]);

      return false;

    }

    if (!satisfy_constrains(i, num_dimension))

    {

      printf("alpha[%d] not satisfy constrains\n", i);

      return false;

    }

  }



  if (fabs(sum) > eps)

  {

    printf("Sum = %lf\n", sum);

    return false;

  }

  return true;

}

/*

min 1/2*||w||^2

s.t.  (w[i]*x[i] + b[i] - y[i]) >= 0;

*/

/*

step 1: cal alpha[]

step 2: cal w,b

*/



/*

min(para alpha) 1/2*sum(i)sum(j)(alpha[i]*alpha[j]*y[i]*y[j]*x[i]*x[j]) - sum(alpha[i])

s.t. sum(alpha[i] * y[i]) = 0

C>= alpha[i] >= 0

*/



int main()
{
  scanf("%d%d", &num_samples, &num_dimension);
  for (int i = 0; i < num_samples; i++)
  {
    for (int j = 0; j < num_dimension; j++)
    {
      scanf("%lf", &x[i][j]);
    }
    scanf("%lf", &y[i]);
  }
  c = 1;
  //初值附为0；
  for (int i = 0; i < num_samples; i++)
  {
    alpha[i] = 0;
  }
  int count = 0;
  while (optimize()) {
    calB();
    count++;
  }
  printf("%d ", count);
  calW();
  calB();
  printf("y = ");
  for (int i = 0; i < num_dimension; i++)
  {
    printf("%lf * x[%d] + ", w[i], i);
  }
  printf("%lf\n", b);
  if (!check())
    printf("Not satisfy KKT.\n");
  else
    printf("Satisfy KKT\n");
}



/*

3 2

3 3 1

4 3 1

1 1 -1

*/


