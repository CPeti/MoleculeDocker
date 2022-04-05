//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Czumbel Peter
// Neptun : ODZF0M
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include <time.h>

int maxMass = 50;
int maxCharge = 100;
int bondRadius = 10; //px
float pi = 3.141592653589;



// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

// 2D camera
class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(200, 200) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;
GPUProgram gpuProgram; // vertex and fragment shaders


class Atom {
public:
	int ID = 0;
	int mass;
	int charge;
	int bonds;
	std::vector<Atom*> bondedWith;
	vec2 offset = NULL;
	bool assigned = false;

	unsigned int vao2;
	const int nv = 20;
	float phi = 0;		//angle of rotation
	float sx = 10;
	float sy = 10;		//scaling
	vec2 wTranslate;

	Atom() {
		mass = rand() % maxMass + 1;		//1-50x a hidrogén tömege
		bonds = 0;
		charge = 0;
	}
	void bind(Atom* a) {
		bondedWith.push_back(a);
		bonds++;
	}
	void create() {
		glGenVertexArrays(1, &vao2);
		glBindVertexArray(vao2);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		const int nv = 20;
		vec2 vertices[nv];
		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * pi / nv;
			vertices[i] = vec2(offset.x + 0.3*cosf(fi), offset.y + 0.3*sinf(fi));
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			nv * sizeof(vec2),
			vertices,
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			0, NULL
		);
	}
	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}
	void draw() {
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");
		gpuProgram.setUniform(charge>=0 ? vec3(1.0f, 0.0f, 0.0f) : vec3(0.0f, 0.0f, 1.0f), "color");
		glBindVertexArray(vao2);
		glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
	}

};

class Molecule {
private:
	int size;
	int totalMass = 0;
	std::vector<Atom*> atoms;
	std::vector<vec2> vertices;
	std::vector<vec2> edges;


	vec2 center;		//center of mass
	float phi = 0;		//angle of rotation
	vec2 wTranslate;	//translation
	float sx = 10;
	float sy = 10;		//scaling
	unsigned int vao;
public:
	Molecule() {
		srand(time(NULL));
		size = rand() % 7 + 2;
		for (int i = 0; i < size; i++) {
			atoms.push_back(new Atom());
			atoms[i]->ID = i;
			if (i > 0) {
				int j = rand() % i;
				atoms[i]->bind(atoms[j]);
				atoms[j]->bind(atoms[i]);
			}
		}
		for (int i = 0; i < size; i++) {
			totalMass += atoms[i]->mass;
			int delta = rand() % maxCharge + 1;
			atoms[i]->charge -= delta;
			atoms[(i + 1) % size]->charge += delta;
		}
		int rootIndex = 0;							//node with maximum edges
		int maxBonds = 0;
		for (int i = 0; i < size; i++) {
			if (atoms[i]->bonds > maxBonds) {
				maxBonds = atoms[i]->bonds;
				rootIndex = i;
			}
		}
		atoms[rootIndex]->offset = vec2(0, 0);
		atoms[rootIndex]->assigned = true;
		printf("%d\n", size);
		build(*atoms[rootIndex]);
		for (int i = 0; i < size; i++) {
			vertices.push_back(atoms[i]->offset);
		}

		for (int i = 0; i < size; i++) {
			center = center + atoms[i]->offset * atoms[i]->mass;
		}
		center = center / totalMass;
	}
	void build(Atom root) {
		for (int i = 0; i < root.bonds; i++) {
			if (!root.bondedWith[i]->assigned) {
				float angle = 2 * pi / (root.bonds) + rand() % 180 * pi / 180;
				float x = 5.0f - (0.1 * (float)(rand() % 50) - 2.5);
				float y = 0;
				root.bondedWith[i]->offset = root.offset + vec2(x * cosf(angle * i) - y * sinf(angle * i), x * sinf(angle * i) + y * cosf(angle * i));
				edges.push_back(root.offset);
				edges.push_back(root.bondedWith[i]->offset);
				root.bondedWith[i]->assigned = true;
				build(*root.bondedWith[i]);
			}
		}
	}
	void create() {

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			edges.size() * sizeof(vec2),
			&edges[0],
			GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT,
			GL_FALSE,
			sizeof(vec2), NULL
		);

		for (int i = 0; i < size; i++) {
			atoms[i]->create();
		}
	}

	void update(float t) {
		phi = t;
		for(int i = 0; i < size; i++){
			atoms[i]->phi = t;
		}
	}

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phi), sinf(phi), 0, 0,
			-sinf(phi), cosf(phi), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	void draw() {
		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");
		gpuProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_LINES, 0, edges.size());	// draw edges

		for (int i = 0; i < size; i++) {
			atoms[i]->draw();
		}
	}
	~Molecule() {
	}
};

Molecule* m1 = new Molecule();

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);
	glPointSize(10.0f);
	m1->create();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color <- should be in init()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	m1->draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case 's': camera.Pan(vec2(-1, 0)); break;
		case 'd': camera.Pan(vec2(+1, 0)); break;
		case 'e': camera.Pan(vec2(0, 1)); break;
		case 'x': camera.Pan(vec2(0, -1)); break;
		case 'z': camera.Zoom(0.9f); break;
		case 'Z': camera.Zoom(1.1f); break;
		case ' ': {
			delete m1;
			m1 = new Molecule();
			m1->create();
		}break;
	}
	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;
	m1->update(sec);
	glutPostRedisplay();

}