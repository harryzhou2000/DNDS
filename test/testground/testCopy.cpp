#include <iostream>
#include <vector>
struct AA
{
    int a;
    // void operator=(const AA &other) = delete;
};

struct A
{
    int a;
    std::vector<int> B;
    AA AAinstance;

    A() : a(), B(), AAinstance() {}

    A(const A &other) : a(other.a), B(other.B), AAinstance(other.AAinstance)
    {
        std::cout << "A:copy ctor" << std::endl;
    }

    void operator=(const A &other)
    {
        a = other.a;
        B = other.B;
        AAinstance = other.AAinstance;
        std::cout << "A:copy operator=" << std::endl;
    }

    void operator=(A &&other)
    {
        a = std::move(other.a);
        B = std::move(other.B);
        AAinstance = std::move(other.AAinstance);
        std::cout << "A:move operator=" << std::endl;
    }
};

int main()
{
    A a1;
    a1.a = 1;
    a1.B.resize(2);
    a1.AAinstance.a = 3;

    A a2{a1};

    A a3{a1};

    A a4;

    a4 = std::move(a3);



    std::cout << a1.B.size() << " " << a2.B.size() << std::endl;
    std::cout << a1.B.data() << " " << a2.B.data() << std::endl;

    std::cout << a3.B.size() << " " << a4.B.size() << std::endl;
    std::cout << a3.B.data() << " " << a4.B.data() << std::endl;
    return 0;
}

