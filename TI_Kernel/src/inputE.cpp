//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Fatima-Ezzahra Moukhtari

#include <src/inputE.h>

namespace hfp3d {

    il::Array<double> Cmatrix( const std::string a ) {
        //Read the elastic parameters and the mesh
        std::ifstream i(a);
        json j;
        i >> j;

        //create the stiffness matrix

        il::Array<double> Ce{5};
        int ii=0;
        for (json::iterator it = j["Solid parameters"].begin(); it != j["Solid parameters"].end(); ++it) {

            Ce[ii]=it.value();
            ii++;
        }
        return Ce;
    }

}



