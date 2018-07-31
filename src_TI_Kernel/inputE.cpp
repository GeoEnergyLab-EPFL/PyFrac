



#include "inputE.h"
#include <vector>
#include <iostream>
using namespace std;
namespace hfp3d {


    //Create a json file
    // json o;
    // o["C11"] = 123;
    // o["C13"] = 5;
    // o["C33"] = 3.141;


    std::vector<double> Cmatrix( const std::string a) {
        //Read the elastic parameters and the mesh
        std::ifstream i(a);
        json j;
        i >> j;
        //save a json file
        // std::ofstream l("data.json");
        //  l << std::setw(4) << o << std::endl;


//create the stiffness matrix


        int ii=0;
        std::vector<double> Ce;
        Ce.resize(5);
        for (json::iterator it = j["Solid parameters"].begin(); it != j["Solid parameters"].end(); ++it) {
            Ce[ii]=it.value();
            std::cout <<  "Cij = " << Ce[ii]<< "\n" ;

            ii++;
        }

         return Ce;
    }

}



