#ifndef DATAIO_H__
#define DATAIO_H__

#include <fstream>

template <typename T>
bool saveArr(std::string path, T* arr, int arr_len) {
    std::ofstream FILE(path, std::ios::out | std::ofstream::binary);
    if (!FILE.is_open())
        return false;
    FILE.write(reinterpret_cast<const char *>(arr), arr_len*sizeof(T));
    FILE.close();
    return true;
}

template <typename T>
bool loadArr(std::string path, T* arr, int arr_len) {
    std::ifstream FILE(path, std::ios::in | std::ifstream::binary);
    if (!FILE.is_open())
        return false;
    FILE.read(reinterpret_cast<char *>(arr), arr_len*sizeof(T));
    FILE.close();
    return true;
}

#endif
