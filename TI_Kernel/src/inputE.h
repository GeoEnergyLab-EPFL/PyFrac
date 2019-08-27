//
// Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
// Geo-Energy Laboratory, 2016-2019.  All rights reserved.
// See the LICENSE.TXT file for more details.
//
// Contributors:
// Fatima-Ezzahra Moukhtari

#ifndef HFPX3D_VC_INPUT_H
#define HFPX3D_VC_INPUT_H
#include <fstream>
#include <il/Array.h>
#include <src/nlohmann/json.hpp>
#include <il/String.h>

using json= nlohmann::json;
namespace hfp3d {
    il::Array<double> Cmatrix( const std::string a );
}

#endif
