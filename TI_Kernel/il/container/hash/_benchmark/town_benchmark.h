#pragma once

#include <il/Array.h>
#include <il/Map.h>
#include <il/String.h>
#include <il/Timer.h>
#include <il/print.h>

#include <string>
#include <unordered_map>

inline void hash() {
  const il::int_t nb_expected_cities = 36360;
  const il::int_t nb_times = 4000;
  il::Array<il::String> city{};
  il::Array<std::string> stdcity{};
  il::Array<il::int_t> population{};
  city.Reserve(36360);
  stdcity.Reserve(36360);
  population.Reserve(36360);

  const il::String filename = "/home/fayard/Desktop/villes.txt";
  std::FILE *file = std::fopen(filename.asCString(), "rb");
  const il::int_t max_length = 200;
  il::Array<char> buffer_line{max_length + 1};
  while (std::fgets(buffer_line.Data(), max_length + 1, file) != nullptr) {
    il::String line{il::StringType::Byte, buffer_line.data(),
                    il::size(buffer_line.data())};
    il::StringView v = line.view();
    const il::int_t begin_name = v.nextChar(0, '"') + 1;
    const il::int_t end_name = v.nextChar(begin_name, '"');
    const il::int_t begin_population = v.nextDigit(end_name);
    il::String my_city{v.subview(begin_name, end_name)};
    std::string my_stdcity{my_city.asCString()};
    il::int_t my_population = std::stoi(v.data() + begin_population);
    city.Append(my_city);
    stdcity.Append(my_stdcity);
    population.Append(my_population);
  }
  std::fclose(file);

  il::Timer timer{};
  timer.Start();
  for (il::int_t k = 0; k < nb_times; ++k) {
    il::Map<il::String, il::int_t> map{nb_expected_cities};
    for (il::int_t i = 0; i < city.size(); ++i) {
      map.Set(city[i], population[i]);
    }
  }
  timer.Stop();

  il::print("Time per HashTable creation: {} s\n", timer.time() / nb_times);

  timer.Reset();
  timer.Start();
  for (il::int_t k = 0; k < nb_times; ++k) {
    std::unordered_map<std::string, il::int_t> map{nb_expected_cities};
    for (il::int_t i = 0; i < stdcity.size(); ++i) {
      map[stdcity[i]] = population[i];
    }
  }
  timer.Stop();

  il::print("Time per HashTable creation: {} s\n", timer.time() / nb_times);
}
