/*
 Copyright (c) 2019 CNRS
 Nicolas Bonneel <nicolas.bonneel@liris.cnrs.fr>
 David Coeurjolly <david.coeurjolly@liris.cnrs.fr>
 
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
#include <iostream>
#include <vector>
#include "UnbalancedSliced.h"
#include "CImg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef float FLOAT;

int main()
{
  omp_set_nested(0);
  
  UnbalancedSliced sliced;
  const int nBslices = 30;
  
  //Loading the images
  int W1, H1, W2, H2, comp;
  unsigned char* image1 = stbi_load("data/imageA.jpg", &W1, &H1, &comp, 3);
  if(!image1){
    std::cerr<<"I/O error, input image."<<std::endl;
    exit(2);
  }
  unsigned char* image2 = stbi_load("data/imageB-larger.jpg", &W2, &H2, &comp, 3);
  if(!image2){
    std::cerr<<"I/O error, target image."<<std::endl;
    exit(2);
  }
  std::cout<<"Input image: "<<W1<<"x"<<H1<<std::endl;
  std::cout<<"Target image: "<<W2<<"x"<<H2<<std::endl;
  const int n1 = W1*H1;
  const int n2 = W2*H2;
  const int nbary = n1;
  std::vector<float> weight(2);
  weight[0] = 0;
  weight[1] = 1;
  if (n1 > n2)
  {
    std::cerr<<"The first image must be smaller than the second one."<<std::endl;
    exit(2);
  }
  
  //Creating the diracs
  std::vector<std::vector<Point<3, FLOAT> > > points(2);
  points[0].resize(W1*H1);
  points[1].resize(W2*H2);
  for (int i = 0; i < W1*H1; i++) {
    points[0][i][0] = image1[i * 3] ;
    points[0][i][1] = image1[i * 3+1] ;
    points[0][i][2] = image1[i * 3+2] ;
  }
  for (int i = 0; i < W2*H2; i++) {
    points[1][i][0] = image2[i * 3] ;
    points[1][i][1] = image2[i * 3 + 1] ;
    points[1][i][2] = image2[i * 3 + 2] ;
  }
  
  std::vector<Point<3, FLOAT> > bary(nbary);
  
  //Main computation
  auto start = std::chrono::system_clock::now();
  sliced.correspondencesNd<3, FLOAT>(points[0], points[1], nBslices, true);
  bary = points[0];
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  
  
  //Regularization of the transport plan (optional)
  // (bilateral filter of the difference)
  std::vector<FLOAT> resultdiff(W1*H1 * 3);
  cimg_library::CImg<float> img1float(W1, H1, 1, 3);
  for (int i = 0; i < W1*H1; i++) {
    Point<3, FLOAT> col = bary[i];
    resultdiff[i] = col[0] - points[0][i][0];
    resultdiff[i + W1 * H1] = col[1] - points[0][i][1];
    resultdiff[i + W1 * H1 * 2] = col[2] - points[0][i][2];
    img1float[i] = points[0][i][0];
    img1float[i + W1 * H1] = points[0][i][1];
    img1float[i + W1 * H1 * 2] = points[0][i][2];
  }
  
  cimg_library::CImg<float> image(&resultdiff[0], W1, H1, 1, 3);
  image.blur_bilateral(img1float, 20, 10);
  
  for (int i = 0; i < W1*H1; i++) {
    image1[i * 3] = std::min(255, std::max(0, (int)image.data()[i] + (int)points[0][i][0]));
    image1[i * 3 + 1] = std::min(255, std::max(0, (int)image.data()[i + W1 * H1] + (int)points[0][i][1]));
    image1[i * 3 + 2] = std::min(255, std::max(0, (int)image.data()[i + W1 * H1 * 2] + (int)points[0][i][2]));
  }
  
  //Export
  stbi_write_png("outtransfer.png", W1, H1, 3, &image1[0], 0);
  
  return 0;
}
