#ifndef DEVICEPROPERTIES_H
#define DEVICEPROPERTIES_H

//print colors
#define RESET "\033[0m"
#define RED "\x1B[31m"
#define GREEN "\x1B[32m"
#define YELLOW "\x1B[33m"
#define BLUE "\x1B[34m"
#define MAGETA "\x1B[35m"
#define CYNA "\x1B[36m"
#define WHITE "\x1B[37m"

void printDevProp(cudaDeviceProp devProp);

#endif