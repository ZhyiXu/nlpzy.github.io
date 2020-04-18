```c++
namespace efanna2e {

class IndexNSG : public Index {
 public:
  //explicit用来修饰类的构造函数，被修饰的侯高函数的类，不能发生相应的隐式类型转换，只能以显示方式进行类型转换
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);
 //virtual涉及继承和多台量大特性。虚函数是指类中你希望重载的成员函数，当你用一个基类指针或引用指向一个继承类对象的时候，你调用一个虚函数，实际调用的是继承类的版本。？？？

  virtual ~IndexNSG();

  virtual void Save(const char *filename)override; //保证重载正确性？从名字，到形参再到返回值所有检查
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices);
  void OptimizeGraph(float* data);

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
};
}
```

阅读上面.h代码了解C++各种关键字

首先理解explicit

```c++
//在C++中，explicit关键字用来修饰类的构造函数，被修饰的构造函数的类，不能发生相应的隐式类型转换

//例子：

//未加explicit时的隐式类型转换

   1. class Circle  
   2. {  
   3. public:  
   4.     Circle(double r) : R(r) {}  
   5.     Circle(int x, int y = 0) : X(x), Y(y) {}  
   6.     Circle(const Circle& c) : R(c.R), X(c.X), Y(c.Y) {}  
   7. private:  
   8.     double R;  
   9.     int    X;  
  10.     int    Y;  
  11. };  
  12.   
  13. int _tmain(int argc, _TCHAR* argv[])  
  14. {  
  15. //发生隐式类型转换  
  16. //编译器会将它变成如下代码  
  17. //tmp = Circle(1.23)  
  18. //Circle A(tmp);  
  19. //tmp.~Circle();  
  20.     Circle A = 1.23;   
  21. //注意是int型的，调用的是Circle(int x, int y = 0)  
  22. //它虽然有2个参数，但后一个有默认值，任然能发生隐式转换  
  23.     Circle B = 123;  
  24. //这个算隐式调用了拷贝构造函数  
  25.     Circle C = A;  
  26.       
  27.     return 0;  
  28. } 

加了explicit关键字后，可防止以上隐式类型转换发生

   1. class Circle  
   2. {  
   3. public:  
       	  //Circle类名，double r形参，R(r)r初始值
   4.     explicit Circle(double r) : R(r) {}  
   5.     explicit Circle(int x, int y = 0) : X(x), Y(y) {}  
   6.     explicit Circle(const Circle& c) : R(c.R), X(c.X), Y(c.Y) {}  
   7. private:  
   8.     double R;  
   9.     int    X;  
  10.     int    Y;  
  11. };  
  12.   
  13. int _tmain(int argc, _TCHAR* argv[])  
  14. {  
  15. //一下3句，都会报错  
  16.     //Circle A = 1.23;   
  17.     //Circle B = 123;  
  18.     //Circle C = A;  
  19.       
  20. //只能用显示的方式调用了  
  21. //未给拷贝构造函数加explicit之前可以这样  
  22.          Circle A = Circle(1.23);  
  23.         Circle B = Circle(123);  
  24.         Circle C = A;  
  25.   
  26. //给拷贝构造函数加了explicit后只能这样了  
  27.          Circle A(1.23);  
  28.         Circle B(123);  
  29.         Circle C(A);  
  30.     return 0;  
  31. } 
```

这里又带来一个问题，类内函数声明是怎么样

```c++
Sale_item(const double&re,unsigned  &us_sol):units_sold(us_sol),revenue(re){}
Sale_item：类名；
const double&re,unsigned  &us_sol 形参1，形参2
units_sold(us_sol),revenue(re) 成员名1（初值us_sol），成员名2（初值re）
```

virtual

```C++
#include "stdio.h" 
#include "conio.h"
class Parent
{	
public:
	
	char data[20];
	void Function1();	
	virtual void Function2();   // 这里声明Function2是虚函数
	
}parent;
 
void Parent::Function1()
{
	printf("This is parent,function1\n");
}
 
void Parent::Function2()
 
{
	printf("This is parent,function2\n");
}
 
class Child:public Parent //公有继承，child继承了parent
 
{
	void Function1();
	void Function2();
	
} child;
 
void Child::Function1()
 
{
	printf("This is child,function1\n");
}
 
void Child::Function2()
 
{
	printf("This is child,function2\n");
}
 
int main(int argc, char* argv[])
 
{
	Parent *p;  // 定义一个基类指针
	if(_getch()=='c')     // 如果输入一个小写字母c	
		p=&child;         // 指向继承类对象
	else	
		p=&parent;       // 否则指向基类对象
	p->Function1();   // 这里在编译时会直接给出Parent::Function1()的入口地址。	
	p->Function2();    // 注意这里，执行的是哪一个Function2？
	return 0;	
}
```

如果输入字符c则输出

```
This is parent,function1
This is child,function2
因为是用一个parent类的指针调用函数funtion1，虽然实际上这个指针指向的是child对象，但是编译器没办法知道这个事实，只能按照调用parent类函数来理解并编译。第二行是因为function2被virtual关键字修饰，是一个虚函数，可以动态联编，可以在运行时判断指针指向的对象，并自动调用相应的函数。
```

如果输入非c字符则输出

```
This is parent,function1
This is parent,function2
第二行的变化，根据用户的输入，调整到底调用基类中的function2还是继承类中的function2。这就是虚函数的作用。在有些场景下，重载使用函数时，虚函数就很重要，可以保证用户一直使用的是自己编写的函数，而不是继承类。
```

*解释一下公有继承

公有继承时基类中各成员属性保持不变，基类中private成员被隐藏。派生类的成员只能访问基类中的public/protected成员，而不能访问private成员；派生类的对象只能访问基类中的public成员。
私有继承时基类中各成员属性均变为private，并且基类中private成员被隐藏。派生类的成员也只能访问基类中的public/protected成员，而不能访问private成员；派生类的对象不能访问基类中的任何的成员。 
保护继承时基类中各成员属性均变为protected，并且基类中private成员被隐藏。派生类的成员只能访问基类中的public/protected成员，而不能访问private成员；派生类的对象不能访问基类中的任何的成员。

```
//公有继承                      对象访问    成员访问
public    -->  public              Y         Y
protected -->  protected           N         Y
private   -->  private             N         N
 
//保护继承                      对象访问    成员访问
public    -->  protected           N         Y
protected -->  protected           N         Y
private   -->  protected           N         N
 
//私有继承                      对象访问    成员访问
public    -->  private             N         Y
protected -->  private             N         Y
private   -->  private             N         N
```

------

##### 构造函数与析构函数

C++提供了构造函数来处理对象的初始化。构造函数是一种特殊的成员函数，与其他成员函数不同，构造函数不需要用户来调用它，而是建立对象时自动执行。 （构造函数名与类名相同）

```c++
#include <iostream>
using namespace std;
class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line();  // 这是构造函数
 
   private:
      double length;
};
// 成员函数定义，包括构造函数
Line::Line(void)
{
    cout << "Object is being created" << endl;
}
void Line::setLength( double len )
{
    length = len;
}
double Line::getLength( void )
{
    return length;
}
// 程序的主函数
int main( )
{
   Line line; 
   // 设置长度
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
 
   return 0;
}

```

类的**析构函数**是类的一种特殊的成员函数，它会在每次删除所创建的对象时执行。

析构函数的名称与类的名称是完全相同的，只是在前面加了个波浪号（~）作为前缀，**它不会返回任何值，也不能带有任何参数。析构函数有助于在跳出程序（比如关闭文件、释放内存等）前释放资源**。

```c++
#include <iostream>
using namespace std;
class Line
{
   public:
      void setLength( double len );
      double getLength( void );
      Line();   // 这是构造函数声明
      ~Line();  // 这是析构函数声明
 
   private:
      double length;
};
// 成员函数定义，包括构造函数
Line::Line(void)
{
    cout << "Object is being created" << endl;
}
Line::~Line(void)
{
    cout << "Object is being deleted" << endl;
}
void Line::setLength( double len )
{
    length = len;
}
double Line::getLength( void )
{
    return length;
}
// 程序的主函数
int main( )
{
   Line line;
   // 设置长度
   line.setLength(6.0); 
   cout << "Length of line : " << line.getLength() <<endl;
   return 0;
}
//执行结果
Object is being created
Length of line : 6
Object is being deleted
```

------

##### inline

在 c/c++ 中，为了解决一些频繁调用的小函数大量消耗栈空间（栈内存）的问题，特别的引入了 **inline** 修饰符，表示为内联函数。

栈空间就是指放置程序的局部数据（也就是函数内数据）的内存空间。

在系统下，栈空间是有限的，假如频繁大量的使用就会造成因栈空间不足而导致程序出错的问题，如，函数的死循环递归调用的最终结果就是导致栈内存空间枯竭。