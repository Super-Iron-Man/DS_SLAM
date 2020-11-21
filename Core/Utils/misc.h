#ifndef __MISC_H__
#define __MISC_H__

#include <vector>
#include "Utils/num_type.h"
#include "Utils/settings.h"

namespace ds_slam
{

template <typename T>
inline void DeleteOut(std::vector<T *> &v, const int i)
{
    delete v[i];
    v[i] = v.back();
    v.pop_back();
}

template <typename T>
inline void DeleteOutPt(std::vector<T *> &v, const T *i)
{
    delete i;

    for (unsigned int k = 0; k < v.size(); k++)
        if (v[k] == i)
        {
            v[k] = v.back();
            v.pop_back();
        }
}

template <typename T>
inline void DeleteOutOrder(std::vector<T *> &v, const int i)
{
    delete v[i];
    for (unsigned int k = i + 1; k < v.size(); k++)
        v[k - 1] = v[k];
    v.pop_back();
}

template <typename T>
inline void DeleteOutOrder(std::vector<T *> &v, const T *element)
{
    int i = -1;
    for (unsigned int k = 0; k < v.size(); k++)
    {
        if (v[k] == element)
        {
            i = k;
            break;
        }
    }
    assert(i != -1);

    for (unsigned int k = i + 1; k < v.size(); k++)
        v[k - 1] = v[k];
    v.pop_back();

    delete element;
}

inline bool EigenTestNan(const MatXX &m, std::string msg)
{
    bool foundNan = false;
    for (int y = 0; y < m.rows(); y++)
        for (int x = 0; x < m.cols(); x++)
        {
            if (!std::isfinite((double)m(y, x)))
                foundNan = true;
        }

    if (foundNan)
    {
        printf("NAN in %s:\n", msg.c_str());
        std::cout << m << "\n\n";
    }

    return foundNan;
}

inline void handleKey(char k)
{
    char kkk = k;
    switch (kkk)
    {
    case 'd':
    case 'D':
        freeDebugParam5 = ((int)(freeDebugParam5 + 1)) % 10;
        printf("new freeDebugParam5: %f!\n", freeDebugParam5);
        break;
    case 's':
    case 'S':
        freeDebugParam5 = ((int)(freeDebugParam5 - 1 + 10)) % 10;
        printf("new freeDebugParam5: %f!\n", freeDebugParam5);
        break;
    }
}

}

#endif