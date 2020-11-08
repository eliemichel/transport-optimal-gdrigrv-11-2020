/*
 Copyright (c) 2020 CNRS
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * This is a version of the base file by Elie Michel
 * Original file can be found at:
 * https://github.com/dcoeurjo/transport-optimal-gdrigrv-11-2020/blob/main/c%2B%2B/slicedOT/colorTransfer.cpp
 */

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

//Command-line parsing
#include "CLI11.hpp"

//Image filtering and I/O
#define cimg_display 0
#include "CImg.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Turn this to 0 to avoid using parallelization for batching
// NB batches are useless if not parallelized (actually somehow
// worst than sequential in that case) so turning this off must
// only be for debugging purposes
#define PARALLEL_BATCH 1

//Global flag to silent verbose messages
bool silent;

/**
 * A utility class to measure performance timings
 * Usage: t = BenchmarkTimer(); ...; cout << "it took " << t.ellapsed() << " ms";
 */
class BenchmarkTimer
{
public:
    BenchmarkTimer() {
        reset();
    }

    /**
     * By default the timer starts at creation but it may be reset at some point
     * using this method.
     */
    void reset() {
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    template <typename Duration>
    static double milliseconds(Duration dt)
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(dt).count();
    }

    /**
     * Return the precise ellapsed system time since creation of this object in milliseconds
     */
    double ellapsed() const
    {
        return milliseconds(std::chrono::high_resolution_clock::now() - m_startTime);
    }

private:
    std::chrono::high_resolution_clock::time_point m_startTime;
};

/**
 * A basic Vector3 type used here represent colors
 */
class Vector3 {
public:
    using real = double;
public:
    /**
     * Return a random direction evenly distributed on the unit sphere
     * restricted to the directions that are orthogonal to the previously
     * drawn one (not garranteed in parallel execution).
     */
    static Vector3 DrawRandomDirection()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<real> distrib;
        
        // simple heuristic: we remember the previous direction to
        // always sample a new one otrhogonal to it.
        static Vector3 previous(0, 0, 1);
        
        Vector3 d = Vector3(distrib(gen), distrib(gen), distrib(gen));
        d = d - previous * d.dot(previous);
        d = d.normalized();
        previous = d;
        return d;
    }

public:
    Vector3() : x(0), y(0), z(0) {}
    Vector3(real x, real y, real z) : x(x), y(y), z(z) {}

    real norm() const {
        return std::sqrt(dot(*this));
    }

    Vector3 normalized() const {
        real scale = 1.f / norm();
        return *this * scale;
    }

    real dot(const Vector3 & other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vector3 operator*(real scale) const {
        return Vector3(x * scale, y * scale, z * scale);
    }

    Vector3 operator+(const Vector3 & other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    void operator*=(real scale) {
        x *= scale;
        y *= scale;
        z *= scale;
    }

    void operator+=(const Vector3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    void operator-=(const Vector3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    friend std::ostream & operator<<(std::ostream& os, const Vector3 & v) {
        os << "Vector(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }
    
public:
    real x, y, z;
};

/**
 * Simple object oriented wrapper around stb_image library for loading/saving
 * and perform simple pixelwise operations in parallel.
 * NB: This class does not prevent implicit copying of image content so avoid
 * doing imageA = imageB too much, prefer in-place operations such as += over
 * other ones (e.g. +).
 */
class Image
{
public:
    Image(const std::string & filename, const std::string& name = "")
    {
        int nbChannels;
        unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nbChannels, 0);
        if (!silent) std::cout << name << ": " << width << "x" << height << "   (" << nbChannels << ")" << std::endl;

        if (nbChannels < 3)
        {
            std::cout << "Input images must be RGB images." << std::endl;
            exit(1);
        }

        // Convert to doubles
        pixels.resize(width * height);
        for (int i = 0; i < pixels.size(); ++i) {
            int offset = nbChannels * i;
            pixels[i].x = static_cast<double>(data[offset + 0]) / 255.0;
            pixels[i].y = static_cast<double>(data[offset + 1]) / 255.0;
            pixels[i].z = static_cast<double>(data[offset + 2]) / 255.0;
        }
        stbi_image_free(data);
    }

    Image(int width, int height)
        : width(width)
        , height(height)
    {
        pixels.resize(width * height);
    }

    //Image(const Image& other) = delete;
    //Image & operator=(const Image&) = delete;

    const Vector3 & colorAt(int i, int j) const {
      return pixels[width * j + i];
    }

    void write(const std::string& filename) const {
        int nbChannels = 3;
        std::vector<unsigned char> data(nbChannels * width * height);
        for (int i = 0; i < pixels.size(); ++i) {
            const Vector3& p = pixels[i];
            int offset = nbChannels * i;
            data[offset + 0] = static_cast<unsigned char>(std::min(std::max(0.0, p.x * 255.0), 255.0));
            data[offset + 1] = static_cast<unsigned char>(std::min(std::max(0.0, p.y * 255.0), 255.0));
            data[offset + 2] = static_cast<unsigned char>(std::min(std::max(0.0, p.z * 255.0), 255.0));
        }
        if (!silent) std::cout << "Exporting.." << std::endl;
        int errcode = stbi_write_png(filename.c_str(), width, height, nbChannels, data.data(), nbChannels * width);
        if (!errcode)
        {
            std::cout << "Error while exporting the resulting image." << std::endl;
            exit(errcode);
        }
    }

    void operator=(const Vector3 & value) {
        fill(pixels.begin(), pixels.end(), value);
    }

    void operator=(const double value) {
        fill(pixels.begin(), pixels.end(), Vector3(value, value, value));
    }

    void operator+=(const Image& other) {
#pragma omp parallel for
        for (int i = 0; i < pixels.size(); ++i) {
            pixels[i] += other.pixels[i];
        }
    }

    void operator-=(const Image& other) {
#pragma omp parallel for
        for (int i = 0; i < pixels.size(); ++i) {
            pixels[i] -= other.pixels[i];
        }
    }

    void operator/=(double value) {
        double scale = 1 / value;
#pragma omp parallel for
        for (int i = 0; i < pixels.size(); ++i) {
            pixels[i] *= scale;
        }
    }

    /**
     * Convert the image to CImg library representation, to use CImg's bilateral filter
     * (unfortunately not using the same underlying data layout, but copying is 2 orders
     * of magnitude faster than the filtering itself).
     */
    cimg_library::CImg<Vector3::real> asCimg() const {
        std::vector<Vector3::real> cimg_pixels(3 * pixels.size());
#pragma omp parallel for
        for (int i = 0; i < pixels.size(); ++i) {
            const Vector3& p = pixels[i];
            cimg_pixels[i + pixels.size() * 0] = p.x;
            cimg_pixels[i + pixels.size() * 1] = p.y;
            cimg_pixels[i + pixels.size() * 2] = p.z;
        }
        return cimg_library::CImg<Vector3::real>(cimg_pixels.data(), width, height, 1, 3);
    }

    /**
     * Convert back from CImg
     */
    void fromCimg(const cimg_library::CImg<Vector3::real> & cimg) {
        const Vector3::real * cimg_pixels = cimg.data();
#pragma omp parallel for
        for (int i = 0; i < pixels.size(); ++i) {
            Vector3& p = pixels[i];
            p.x = cimg_pixels[i + pixels.size() * 0];
            p.y = cimg_pixels[i + pixels.size() * 1];
            p.z = cimg_pixels[i + pixels.size() * 2];
        }
    }

public:
    std::vector<Vector3> pixels;
    int width, height;
};

/**
 * A projected pixel is represented by its index in the original pixel vector
 * and its projection (dot product) onto the target axis.
 */
class Projection
{
public:
    int index;
    double value;
};

/**
 * Return a vector of projected pixel information (Projection obects) from image
 * sorted by their projection onto the direction theta.
 */
static std::vector<Projection> sortPixelsAlong(const Image & image, const Vector3 & theta)
{
    std::vector<Projection> projections(image.width * image.height);
#pragma omp parallel for
    for (int i = 0; i < projections.size(); ++i) {
        projections[i].index = i;
        projections[i].value = theta.dot(image.pixels[i]);

        double x = theta.dot(image.pixels[projections[i].index]);
        assert(std::abs(projections[i].value - x) < 1e-5);
    }
    std::sort(projections.begin(), projections.end(), [](const Projection& a, const Projection& b) {
        return a.value < b.value;
    });

#ifndef NDEBUG
    // Sanity check
    for (int i = 0; i < projections.size(); ++i) {
        double x = theta.dot(image.pixels[projections[i].index]);
        assert(std::abs(projections[i].value - x) < 1e-5);
    }
#endif // NDEBUG

    return projections;
}

/**
 * Advects colors from source to target along a random direction and store
 * the result into the output image. The advection step is multiplied by epsilon.
 */
static double advect(Image& output, const Image& source, const Image& target, double epsilon = 1.0)
{
    Vector3 theta = Vector3::DrawRandomDirection();
    if (!silent) std::cout << "Projecting onto random direction " << theta << std::endl;
    const std::vector<Projection> & source_proj = sortPixelsAlong(source, theta);
    const std::vector<Projection> & target_proj = sortPixelsAlong(target, theta);

    double energy = 0;
#pragma omp parallel for
    for (int i = 0; i < source_proj.size(); ++i) {
        if (i > 0) {
            assert(source_proj[i - 1].value <= source_proj[i].value);
            assert(target_proj[i - 1].value <= target_proj[i].value);
        }
        double diff = target_proj[i].value - source_proj[i].value;
        int j = source_proj[i].index;
        output.pixels[j] += theta * diff * epsilon;
#pragma omp atomic
        energy += diff * diff;
    }

    return energy / source_proj.size();
}

/**
 * Export transport plan to an ad-hoc file format for visualisation
 */
void exportTransportPlan(const std::string & filename, const Image & source, const Image& output) {
    if (!silent) std::cout << "Exporting transport plan to " << filename << "..." << std::endl;
    std::ofstream file(filename, std::ios::binary);
    unsigned int s = (unsigned int)source.pixels.size();
    file.write((const char*)&s, sizeof(s));
    std::cout << (unsigned int)source.pixels.size() << std::endl;
    for (int i = 0; i < source.pixels.size(); ++i) {
        const Vector3& sp = source.pixels[i];
        file.write((const char*)&sp, sizeof(sp));
        const Vector3& op = output.pixels[i];
        file.write((const char*)&op, sizeof(op));
    }
    file.close();
}

/**
 * Export all steps of the advections to an ad-hoc file format for visualisation
 */
class CheckpointLogger {
public:
    CheckpointLogger() {}
    ~CheckpointLogger() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }
    CheckpointLogger(const CheckpointLogger&) = delete;
    void operator=(const CheckpointLogger&) = delete;

    void start(const std::string & filename, unsigned int width, unsigned int height) {
        m_width = width;
        m_height = height;
        m_file.open(filename, std::ios::binary);
        m_file.write((const char*)&width, sizeof(width));
        m_file.write((const char*)&height, sizeof(height));
    }

    void checkpoint(const Image & image) {
        if (!m_file.is_open()) return;
        assert(image.pixels.size() == m_width * m_height);
        for (int i = 0; i < image.pixels.size(); ++i) {
            const Vector3& sp = image.pixels[i];
            m_file.write((const char*)&sp, sizeof(sp));
        }
    }

private:
    std::ofstream m_file;
    unsigned int m_width, m_height;
};

int main(int argc, char **argv)
{
  CLI::App app{"colorTransfer"};
  std::string sourceImage;
  app.add_option("-s,--source", sourceImage, "Source image")->required()->check(CLI::ExistingFile);;
  std::string targetImage;
  app.add_option("-t,--target", targetImage, "Target image")->required()->check(CLI::ExistingFile);;
  std::string outputImage = "output.png";
  app.add_option("-o,--output", outputImage, "Output image")->required();
  std::string resizedTargetImage = "";
  app.add_option("-z,--resized-target", resizedTargetImage, "Resized target image output");

  std::string transportPlan = "";
  app.add_option("-p,--transport-plan", transportPlan, "Export transport plan into an adhoc .plan format");
  std::string transportRegularizedPlan = "";
  app.add_option("-q,--regularized-transport-plan", transportRegularizedPlan, "Export transport plan after regularization");
  std::string checkpointLogFilename = "";
  app.add_option("-c,--checkpoint-log", checkpointLogFilename, "Another ad-hoc checkpointing format .ot");
  
  unsigned int nbSteps = 3;
  app.add_option("-n,--nbsteps", nbSteps, "Number of sliced steps (3)");
  unsigned int batchSize = 5;
  app.add_option("-b,--batchsize", batchSize, "Batch size (5)");
  double epsilon = 1.0;
  app.add_option("-e,--epsilon", epsilon, "Epsilon learning rate (1.0)");
  float sigmaS = 1.0f;
  app.add_option("-x,--sigma-s", sigmaS, "Standard deviation over spatial dimensions, set to 0 to disable regularization (1.0)");
  float sigmaR = 1.0f;
  app.add_option("-r,--sigma-r", sigmaR, "Standard deviation over color dimensions (1.0)");
  silent = false;
  app.add_flag("--silent", silent, "No verbose messages");
  CLI11_PARSE(app, argc, argv);
  
  Image source(sourceImage, "Source image");
  Image target(targetImage, "Target image");
  Image output(source);

  // Resize target image if it has a different number of pixels than the source.
  if ((source.width* source.height) != (target.width* target.height))
  {
      if (!silent) std::cout << "size do not match, resizing target..." << std::endl;
      Image resized_target(source.width, source.height);
      if (target.width * target.height < source.width * source.height) {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> distrib(0, static_cast<int>(target.pixels.size()) - 1);
          int i = 0;
          for (; i < target.pixels.size(); ++i) {
              resized_target.pixels[i] = target.pixels[i];
          }
          for (; i < resized_target.pixels.size(); ++i) {
              int j = distrib(gen);
              resized_target.pixels[i] = target.pixels[j];
          }
      }
      else {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<int> distrib(0, static_cast<int>(target.pixels.size()) - 1);
          for (int i = 0; i < resized_target.pixels.size(); ++i) {
              int j = distrib(gen);
              resized_target.pixels[i] = target.pixels[j];
          }
      }
      target = resized_target;
      if (!resizedTargetImage.empty()) {
          target.write(resizedTargetImage);
      }
  }

  // Checkpointing
  CheckpointLogger logger;
  if (!checkpointLogFilename.empty()) {
      logger.start(checkpointLogFilename, source.width, source.height);
  }

  logger.checkpoint(source);
  
  // Allocate memory for all images of a batch because they will be filled in parallel
  // this makes batching more memory intensive (but gives the benefit of running in
  // parallel).
#if PARALLEL_BATCH
  std::vector<std::unique_ptr<Image>> batches(batchSize);
  for (unsigned int k = 0; k < batchSize; ++k) {
      batches[k] = std::make_unique<Image>(source.width, source.height);
  }
#  define BATCH *batches[k]
#else // PARALLEL_BATCH
  Image batch(source.width, source.height);
#  define BATCH batch
#endif // PARALLEL_BATCH

  // Core of Sliced Optimal Transport
  BenchmarkTimer otTimer;
  for (unsigned int i = 0; i < nbSteps; ++i) {
      if (batchSize <= 1) {
          double energy = advect(output, output, target, epsilon);
          if (!silent) std::cout << "energy: " << energy << std::endl;
      }
      else
      {
          if (!silent) std::cout << "starting batch #" << i << "..." << std::endl;
          double batchEnergy = 0;

#if !PARALLEL_BATCH
          BATCH = 0;
#endif // PARALLEL_BATCH

#if PARALLEL_BATCH
#  pragma omp parallel for
#endif // PARALLEL_BATCH
          for (int k = 0; k < (int)batchSize; ++k) {
#if PARALLEL_BATCH
              BATCH = 0;
#endif // PARALLEL_BATCH
              double energy = advect(BATCH, output, target, epsilon);
              if (!silent) std::cout << "energy: " << energy << std::endl;
#if PARALLEL_BATCH
#  pragma omp atomic
#endif // PARALLEL_BATCH
              batchEnergy += energy;
          }

          batchEnergy /= batchSize;
          if (!silent) std::cout << "batch energy: " << batchEnergy << "..." << std::endl;

          for (unsigned int k = 0; k < batchSize; ++k) {
              BATCH /= batchSize;
              output += BATCH;
          }
      }

      logger.checkpoint(output);
  }
  double duration = otTimer.ellapsed();
  if (!silent) std::cout << "Optimal Transport computed in " << duration << " ms" << std::endl;

  // Plan export
  if (!transportPlan.empty()) {
      exportTransportPlan(transportPlan, source, output);
  }
  
  // Regularization using bilateral filtering on transport plan
  if (sigmaS > 0) {
      BenchmarkTimer regTimer, convertTimer;
      double convertDuration = 0;
      Image plan(output);
      plan -= source;

      // We convert to CImg to use their bilateral blur
      convertTimer.reset();
      auto plan_cimg = plan.asCimg();
      auto source_cimg = source.asCimg();
      convertDuration += convertTimer.ellapsed();

      plan_cimg.blur_bilateral(source_cimg, sigmaS, sigmaR);

      // Convert back
      convertTimer.reset();
      plan.fromCimg(plan_cimg);
      convertDuration += convertTimer.ellapsed();

      output = source;
      output += plan;
      double duration = regTimer.ellapsed();
      if (!silent) std::cout << "Regularization computed in " << duration << " ms (including " << convertDuration << " ms for round trip to CImg)" << std::endl;
  }

  if (!transportRegularizedPlan.empty()) {
      exportTransportPlan(transportRegularizedPlan, source, output);
  }

  logger.checkpoint(output);

  // Final export
  if (!silent) std::cout << "Optimal Transport computed in " << duration << " ms" << std::endl;
  output.write(outputImage);
  
  exit(0);
}
