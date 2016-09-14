
#include <stdio.h>
#include<GL/glew.h>
#include<gl/glut.h>
#include<glm/glm.hpp>
#include <thrust/device_vector.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define WIDTH 640
#define HEIGHT 640
#define DIM WIDTH

using namespace glm;
using namespace std;

int imageSize =  WIDTH * HEIGHT * 4 ;
unsigned char    *pixels;
unsigned char   *devPixels;

	vec3* devVec3;
	ivec4* devFaces;


__device__ float kEpsilon = 0.001;
//__device__ triangleMesh mesh;
struct ray{
	__device__
	ray(vec3 o  , vec3 d):origin(o) , direction(d){}
	__device__
		ray():origin(vec3(-1)) , direction(vec3(-1)){}
	vec3 origin;
	vec3 direction;
};
glm::vec3 traceRay(ray r  , vec3* vertices , ivec4* faces , int numOfFaces, int depth );

enum materialType{

Matte  = 0,
Specular ,
Mirror


};

struct rayIntersection{
	vec3 point;
	vec3 color;
	vec3 normal;
	materialType mat;
	ray intersectingRay;
	float t;

	__device__
		rayIntersection():t(1000) ,point(vec3(-1)) , normal(vec3(-1)), color(vec3(-1)), mat(materialType::Matte) {}
	__device__
	rayIntersection(float t , vec3 point , vec3 normal , vec3 color , ray r):t(t) ,point(point) , normal(normal), color(color) , intersectingRay(r) {}

};

struct triangle{
	vec3 v0 , v1 , v2;
	vec3 normal;
	vec3 color;
	materialType material;
	__device__ 
		triangle(vec3 a  , vec3 b , vec3 c , materialType mat = materialType::Matte):v0(a),v1(b) ,v2(c) , material(mat){
			vec3 ab = b - a;
			vec3 bc = c - b;

			normal = normalize(cross(bc , ab));
	
	}

	
__device__
	rayIntersection hit(ray ra){
		rayIntersection intersection;
	float a = v0.x - v1.x, b = v0.x - v2.x, c = ra.direction.x, d = v0.x - ra.origin.x; 
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ra.direction.y, h = v0.y - ra.origin.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ra.direction.z, l = v0.z - ra.origin.z;
		
	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;
	
	float inv_denom  = 1.0 / (a * m + b * q + c * s);
	
	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;
	
	if (beta < 0.0)
	 	return intersection;//(false);
	
	float r = r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;
	
	if (gamma < 0.0 )
	 	return intersection;//(false);
	
	if (beta + gamma > 1.0)
		return intersection;//(false);
			
	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;
	
	//if (t < kEpsilon) 
		//return (false);
				
//	rayIntersection intersection;
	intersection .t 		= t;
	intersection.color = color;
	intersection.normal = normal;
	intersection.point = ra.origin + t * ra.direction;
	intersection.mat = material;
	intersection.intersectingRay = ra;
	return intersection;//(true);	

}
};


struct triangleMesh{

	vec3* vertices;
	ivec4* faces;
	vec3* normal;
	vec3* color;
	vec3 flat_color;
	int numOfFaces;
	__device__
	rayIntersection hit(ray r){
		rayIntersection curIntersection;
		for(int i = 0 ;  i < numOfFaces ; i++){
			triangle tri(vertices[faces[i].x] , vertices[faces[i].y] , vertices[faces[i].z] , (materialType)faces[i].w);
			rayIntersection intersection = tri.hit(r);
			if(curIntersection.t > intersection.t){
				curIntersection = intersection;
			}
		
		}
			return curIntersection;
	
	}
	
	};
	




__device__
glm::mat3 getCamera(glm::vec3 eye , glm::vec3 lookAt , float angle){
	glm::vec3 up = glm::vec3(glm::sin(angle) , glm::cos(angle) ,0);

	glm::vec3 w = glm::normalize(lookAt - eye);
	glm::vec3 u = glm::normalize(glm::cross(up , w));
	glm::vec3 v = glm::normalize(glm::cross(w , u));

	return glm::mat3(u , v , w);


}

__device__ bool checkShadow(ray shadowRay ,  vec3* vertices , ivec4* faces , int numOfFaces){

	for(int i = 0 ;  i < numOfFaces ; i++){
			triangle tri(vertices[faces[i].x] , vertices[faces[i].y] , vertices[faces[i].z] );
			rayIntersection intersection = tri.hit(shadowRay);
			if(intersection.t != 1000 && intersection.t > 0)
			return false;
			}
	return true;
}


__device__ vec3 shaderSurface(rayIntersection curIntersection, vec3* vertices , ivec4* faces , int numOfFaces , int depth ){

	
	vec3 lightPos = vec3(10 , 10 , 10);
	vec3 lightDir = normalize(lightPos - curIntersection.point);
	vec3 color = vec3(0);
	ray shadowRay = ray(curIntersection.point + vec3(.01) , lightDir);

	if(checkShadow( shadowRay ,   vertices ,  faces ,  numOfFaces))
	{
		if(curIntersection.mat == materialType::Mirror){
			ray reflectedRay = ray(curIntersection.point + vec3(.01)  , glm::reflect(normalize(curIntersection.intersectingRay.direction) , curIntersection.normal));
		//	color += traceRay(  reflectedRay , vertices ,  faces ,  numOfFaces , --depth);
		color +=  vec3(1 , 1 , 1);
		}
		else{
	float kd = max(0.0f , dot(curIntersection.normal , lightDir));
	color += kd * vec3(1 ,0, 1);
		}
	}
	return color;

}

__device__
glm::vec3 traceRay(ray r  , vec3* vertices , ivec4* faces , int numOfFaces, int depth ){
	
	if(depth == 0)
		return vec3(0);
	//triangle tri(vec3(-1 , -1  , 0 ) , vec3(1 , -1  , 0 ) , vec3(0 , 1  , 0 ) );
	

	/*
	mesh.vertices.push_back(vec3(-.5 , -.5  ,10 ));
	mesh.vertices.push_back(vec3( .5 , -.5  ,10 ));
	mesh.vertices.push_back(vec3( .5 ,  .5  ,10 ));
	mesh.vertices.push_back(vec3(-.5 ,  .5  ,10 ));

	mesh.faces.push_back(ivec3(0 , 1 , 2));
	mesh.faces.push_back(ivec3(0 , 2 , 3));
	*/

	rayIntersection curIntersection;// = tri.hit(r);
//	numOfFaces =0;
//	rayIntersection curIntersection;
		for(int i = 0 ;  i < numOfFaces ; i++){
			triangle tri(vertices[faces[i].x] , vertices[faces[i].y] , vertices[faces[i].z] , (materialType)faces[i].w);
			rayIntersection intersection = tri.hit(r);
			if(curIntersection.t > intersection.t){
				curIntersection = intersection;
			}
		}

	glm::vec3 unitDirection = glm::normalize(r.direction);
	float t = 0.5 * (unitDirection.y + 1.0);
	vec3 color;
	float tmin = 0.01;
	if(curIntersection.t != 1000 ) color = shaderSurface(curIntersection,  vertices , faces ,  numOfFaces , --depth);//vec3(1);
	else
	color =  glm::vec3(0.5 , 0.7 , 1.0) *  t;

	return color;

}


__global__ void kernel( unsigned char *ptr , vec3* vertices , ivec4* faces , int numOfFaces , float time ) {

	   int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
	float aspectRatio =	float(WIDTH)/float(HEIGHT);
	
	float xx = (2 * (x + 0.5 ) /WIDTH - 1)  * aspectRatio;
	float yy = (2 * ( y + 0.5) / HEIGHT - 1);
	
	glm::vec3 rayOrigin ;//= glm::vec3(0 , 0 , 0);
	float r = 5;
	float theta = 0.0f;
	rayOrigin = glm::vec3(r * glm::sin(time) , 0 , r * glm::cos(time));
	rayOrigin = vec3(0 , 0 , 5);
	glm::vec3 rayDirection = glm::vec3(xx , yy  , 1 );
	glm::mat3 cam = getCamera(rayOrigin , glm::vec3(0 , 0 , 0) , 0.0f);
	rayDirection = cam * rayDirection;

	ray ra(rayOrigin , rayDirection);
	glm::vec3 color = traceRay(ra  , vertices , faces , numOfFaces  ,2  );

	 

	 ptr[offset*4 + 0] = (int)(color.r * 255);
						//(255*(1+rayDirection.x)*0.5);
						//1;
    ptr[offset*4 + 1] = (int)(color.g * 255);
						//(255*(1+rayDirection.y)*0.5);
						//0;
    ptr[offset*4 + 2] = (int)(color.b * 255);
						//0;
    ptr[offset*4 + 3] = 1;//255;

}

triangleMesh *devTriangleMesh;
	triangleMesh* hostTriangleMesh;
void init(){

	
	//cuda setup
	cudaMalloc( (void**)&devPixels,imageSize);
	cudaMemset(devPixels , 0  , imageSize);

	int meshSize = 500;// sizeof(triangleMesh);
	//hostTriangleMesh = (triangleMesh*)malloc(meshSize);
/*
	hostTriangleMesh->vertices = (vec3*)malloc(sizeof(vec3) * 4);
	hostTriangleMesh->faces = (ivec3*)malloc(sizeof(ivec3) * 4);


	hostTriangleMesh->vertices = new vec3[4];
	hostTriangleMesh->faces = new ivec3[2];

	hostTriangleMesh->vertices[0] = vec3(-.5 , -.5  ,10 );
	hostTriangleMesh->vertices[1] = vec3( .5 , -.5  ,10 );
	hostTriangleMesh->vertices[2] = vec3( .5 ,  .5  ,10 );
	hostTriangleMesh->vertices[3] = vec3(-.5 ,  .5  ,10 );

	

	hostTriangleMesh->faces[0] = vec3(0 , 1 , 2);
	hostTriangleMesh->faces[1] = vec3(0 , 2 , 3);

	hostTriangleMesh->numOfFaces = 2;
	*/

	vec3* vertices = new vec3[4];
	vertices[0] = vec3(-.5 , -.5  ,10 );
	vertices[1] = vec3( .5 , -.5  ,10 );
	vertices[2] = vec3( .5 ,  .5  ,10 );
	vertices[3] = vec3(-.5 ,  .5  ,10 );
	//713 //1400
	ivec3* faces = new ivec3[2];
	faces[0] = vec3(0 , 1 , 2);
	faces[1] = vec3(0 , 2 , 3);


	 const int isoFaces[] = {
        2, 1, 0,        0,
        3, 2, 0,		0,
        4, 3, 0,		0,
        5, 4, 0,		0,
        1, 5, 0,		0,
        11, 6,  7,		0,
        11, 7,  8,		0,
        11, 8,  9,		0,
        11, 9,  10,		0,
        11, 10, 6,		0,
        1, 2, 6,		0,
        2, 3, 7,		0,
        3, 4, 8,		0,
        4, 5, 9,		0,
        5, 1, 10,		0,
        2,  7, 6,		0,
        3,  8, 7,		0,
        4,  9, 8,		0,
        5, 10, 9,		0,
        1, 6, 10,		0,
	   12 , 14, 13,		0,
		13 ,  14 , 15,	0,
		12 , 16 ,14,	2,
		14 ,16 ,17,     2,   };

    const float isoVerts[] = {
         0.000f,  0.000f,  1.000f,     //0 
         0.894f,  0.000f,  0.447f,	  
         0.276f,  0.851f,  0.447f,	  
        -0.724f,  0.526f,  0.447f,	  
        -0.724f, -0.526f,  0.447f,	  
         0.276f, -0.851f,  0.447f,	  
         0.724f,  0.526f, -0.447f,	  
        -0.276f,  0.851f, -0.447f,	  
        -0.894f,  0.000f, -0.447f,	  
        -0.276f, -0.851f, -0.447f,	  
         0.724f, -0.526f, -0.447f,	  
         0.000f,  0.000f, -1.000f ,	  
	   -2.00f , -1.000f ,  2.00f,      //12
		2.00f , -1.000f ,  2.00f,	  
	   -2.00f , -1.000f , -2.00f,	  
		2.00f , -1.000f , -2.00f,      //15 
									  
		-2.00f , 1.000f ,  2.00f,      //16
		-2.00f , 1.000f ,  -2.00f,          };//17

		 


	cudaMalloc((void**)&devVec3 , sizeof(vec3)*18);
	cudaMalloc((void**)&devFaces , sizeof(ivec4)*24);

	
	cudaMemcpy( devVec3, isoVerts, sizeof(vec3)*18, cudaMemcpyHostToDevice );
	cudaMemcpy( devFaces, isoFaces, sizeof(ivec4)*24, cudaMemcpyHostToDevice );
	
	//loadFishToDevise(devVec3 , devFaces);
	//	 mesh* fish = new mesh();
/*		 int numOfFishVert = 713;
		 int numOfFishFaces = 1400;
	glm::vec3* fishVertices = new vec3[numOfFishVert];
glm::ivec3* fishFaces = new ivec3[numOfFishFaces];
	loadFish(fishVertices , fishFaces);
	
	//std::cout  << fish->vertices.size() << " " << fish->faces.size() << std::endl;
	cudaMalloc((void**)&devVec3 , sizeof(glm::vec3) * numOfFishVert);
	cudaMalloc((void**)&devFaces , sizeof(glm::ivec3) * numOfFishFaces);

	cudaMemcpy( devVec3, fishVertices,  sizeof(glm::vec3) * numOfFishVert, cudaMemcpyHostToDevice );
	cudaMemcpy( devFaces, fishFaces, sizeof(glm::ivec3) * numOfFishFaces  , cudaMemcpyHostToDevice );
	*/

	glClearColor( 1.0, 1.0, 0.0, 1.0 );
	pixels = new unsigned char[imageSize];
//	memset(&pixels , 255 , WIDTH  * HEIGHT *  4);
//	for(int i = 0 ;  i  < imageSize; i += 4){
//	pixels[i] = 255;
//	}
}
void Draw()
{
	glClear(GL_COLOR_BUFFER_BIT);
	  dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
	float time = 0.0f;//glutGet(GLUT_ELAPSED_TIME)/1000;//0.0f;
    kernel<<<grids,threads>>>( devPixels , devVec3 , devFaces , 24, time);
	cudaMemcpy( pixels, devPixels, imageSize, cudaMemcpyDeviceToHost );
	 glDrawPixels( WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixels );
	 glutSwapBuffers();
	 glutPostRedisplay();
	  //glFlush();

}
int main(int argc , char** argv){

		glutInit( &argc, argv );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( WIDTH, HEIGHT );
        glutCreateWindow( "RayTrace" );

		glewExperimental;
		glewInit();
       // glutKeyboardFunc(Key);
			init();

        glutDisplayFunc(Draw);
        glutMainLoop();

	
}