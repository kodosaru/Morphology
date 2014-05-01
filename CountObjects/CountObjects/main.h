//
//  main.h
//  CountObjects
//
//  Created by Don Johnson on 4/30/14.
//  Copyright (c) 2014 Donald Johnson. All rights reserved.
//

#ifndef CountObjects_main_h
#define CountObjects_main_h

#include <iostream>
#include <fstream>
#include <string>

extern std::ofstream logfile;
bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

#endif
