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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, sqrt(vp.x*vp.x+vp.y*vp.y+1.0f)) * MVP;		// transform vp from modeling space to normalized device space
		//gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
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

vec2 rotate(vec2 point, vec2 pivot, float phi);
float distance(vec2 p1, vec2 p2);


int maxMass = 50;
int maxCharge = 100;
int bondRadius = 10; //px
float pi = 3.141592653589f;
float eCharge = 1.602176634f * pow(10, -1);//-19		//charge of an electron in Coulombs
float hMass = 1.66 *pow(10, -2);//-29			//weight of a hydrogen atom in kg
float epsilon = 8.854187817; //* pow(10, -12);		//vacuum permittivity, As/Vm
float dt = 0.01;
float rho = 1.5;

float wX = 600;
float wY = 600;
// 2D camera
class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(wX, wY) { }

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

	unsigned int vao;
	unsigned int vbo = 0;
	const int nv = 20;
	float phi = 0;

	float phiTranslate = 0;		//angle of rotation
	float sx = wX/2;
	float sy = wY/2;		//scaling
	vec2 wTranslate;
	std::vector<vec2> vertices;
	float r = 0.06;

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
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);

		for (int i = 0; i < nv; i++) {
			float fi = i * 2 * pi / nv;
			float tr = r * (fabs(mass / ((float)maxMass)) + 1)/2;
			vertices.push_back(vec2(offset.x + tr*cosf(fi), offset.y + tr*sinf(fi)));
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			nv * sizeof(vec2),
			&vertices[0],
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

		mat4 Mrotate(cosf(phiTranslate), sinf(phiTranslate), 0, 0,
			-sinf(phiTranslate), cosf(phiTranslate), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}
	void draw() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, nv * sizeof(vec2), &vertices[0]);
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");
		float t = fabs(((float)charge)/ ((float)maxCharge));
		gpuProgram.setUniform(charge>=0 ? vec3(t, 0.0f, 0.0f) : vec3(0.0f, 0.0f, t), "color");
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, nv);
	}
	void move(vec2 t) {
		for (int i = 0; i < nv; i++) {
			vertices[i] = vertices[i] + t;
		}
	}
	void rotateBy(float angle, vec2 pivot) {
		phi += angle;
		offset = rotate(offset, vec2(0, 0), angle);
		for (int i = 0; i < nv; i++) {
			vertices[i] = rotate(vertices[i], pivot, angle);
		}
	}
};

class Molecule {
private:
	int size;						//number of atoms in molecule
	float mass = 0;					//sum of the atoms' mass
	float theta = 0;					

	std::vector<Atom*> atoms;		//stores the atoms
	std::vector<vec2> edges;		//stores edge endpoints
	std::vector<vec2> vEdges;		//stores the vectorized edges
	int nvEdge = 20;				//edges are split into 20 pieces
	vec2 position = vec2(0, 0);		//position of the molecule
	float phi = 0;					//angle of rotation 

	vec3 velocity = (0.0f, 0.0f, 0.0f);			//velocity
	vec3 omega = (0.0f, 0.0f, 0.0f);			//angular velocity
	vec3 force = (0.0f, 0.0f, 0.0f);			//force
	vec3 m = (0.0f, 0.0f, 0.0f);				//torque


	vec2 center;					//center of mass
	float phiTranslate = 0;			//angle of rotation 
	vec2 wTranslate = vec2(0, 0);	//translation
	float sx = wX/2;
	float sy = wY/2;				//scaling
	float timeOfLastUpdate = 0;
	unsigned int vao;
	unsigned int vbo;
public:
	Molecule() {
		//get number of atoms in molecule, between 2 and 8
		size = rand() % 7 + 2;
		//create atoms
		for (int i = 0; i < size; i++) {
			atoms.push_back(new Atom());
			//give an ID to each atom
			atoms[i]->ID = i;
			//connect new atoms to a previous atom, creating a tree
			if (i > 0) {
				int j = rand() % i;
				atoms[i]->bind(atoms[j]);
				atoms[j]->bind(atoms[i]);
			}
		}
		//calculate the total mass of the molecule and set the charges of the atoms
		for (int i = 0; i < size; i++) {
			//add atoms mass to total mass
			mass += atoms[i]->mass;
			//for every atom, substract a random value from its charge and add it to the next atoms charge
			//this way the total charge of the molecule will be 0
			int delta = rand() % maxCharge + 1;
			atoms[i]->charge -= delta;
			atoms[(i + 1) % size]->charge += delta;
		}
		//find the atom with the highest number of bonds, this will be the root of the tree
		int rootIndex = 0;		
		int maxBonds = 0;
		for (int i = 0; i < size; i++) {
			if (atoms[i]->bonds > maxBonds) {
				maxBonds = atoms[i]->bonds;
				rootIndex = i;
			}
		}

		//the roots position will be 0, 0 within the molecule
		atoms[rootIndex]->offset = vec2(0, 0);
		//if the atom has a position assigned, set this to true
		atoms[rootIndex]->assigned = true;
		//set the positions of all other atoms within the molecule
		build(*atoms[rootIndex]);

		//calculate center of mass
		for (int i = 0; i < size; i++) {
			center = center + atoms[i]->offset * atoms[i]->mass;
		}
		center = center / mass;

		//move center of mass to 0, 0 and push all atoms to their proper place
		for (int i = 0; i < size; i++) {
			atoms[i]->offset = atoms[i]->offset - center;
		}

		//move center of mass to 0, 0 and push all edge endpoints to their proper place
		for (int i = 0; i < edges.size(); i++) {
			edges[i] = edges[i] - center;
		}

		//calculate moment of inertia
		for (int i = 0; i < atoms.size(); i++) {
			float r = length(atoms[i]->offset);
			theta += atoms[i]->mass * r * r;						//sum(m * r^2)
		}

		//vectorize the edges
		//split all the edges into 20 smaller pieces, and put the endpoints of thes in vEdges
		for (int i = 0; i < edges.size(); i+=2) {
			for (int j = 0; j < nvEdge; j++) {
				float t = (float)j / (nvEdge-1);
				vEdges.push_back(vec2(edges[i].x * t + edges[i+1].x * (1.0f - t), edges[i].y * t + edges[i + 1].y * (1.0f - t)));
			}
		}
	}
	//sets the position of every atom in the molecule
	//the position is relative to the root atom, at 0, 0
	void build(Atom root) {
		for (int i = 0; i < root.bonds; i++) {
			if (!root.bondedWith[i]->assigned) {
				float angle = pi / 180 * (rand() % 360);
				float l = (0.25f + (float)(rand() % 50) * 0.01f);
				root.bondedWith[i]->offset = root.offset + rotate(vec2(l, 0), vec2(0, 0), angle);

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

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER,
			vEdges.size() * sizeof(vec2),
			&vEdges[0],
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

	mat4 M() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // scaling

		mat4 Mrotate(cosf(phiTranslate), sinf(phiTranslate), 0, 0,
			-sinf(phiTranslate), cosf(phiTranslate), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1); // rotation

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation

		return Mscale * Mrotate * Mtranslate;	// model transformation
	}

	void draw() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, vEdges.size() * sizeof(vec2), &vEdges[0]);
		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		mat4 MVPTransform = M() * camera.V() * camera.P();
		gpuProgram.setUniform(MVPTransform, "MVP");
		gpuProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		for (int i = 0; i < edges.size()/2; i++) {
			glDrawArrays(GL_LINE_STRIP, i*nvEdge, nvEdge);	// draw edges
		}

		for (int i = 0; i < size; i++) {
			atoms[i]->draw();
		}
	}

	void applyForce(Molecule* molecule) {
		omega = vec3(0, 0, 0);
		force = vec3(0, 0, 0);
		m = vec3(0, 0, 0);
		for (int i = 0; i < size; i++) {
			vec3 aVelocity = velocity + cross(omega, atoms[i]->offset);
			//printf("vel: %f, %f, %f\n", aVelocity.x, aVelocity.y, aVelocity.z);
			vec3 aForce = vec3(0, 0, 0);		//air resistance
			for (int j = 0; j < molecule->size; j++) {
				float d = max(distance(position + atoms[i]->offset, molecule->position + molecule->atoms[j]->offset), 0.1);
				aForce = aForce + (atoms[i]->charge * molecule->atoms[j]->charge * pow(eCharge, 2))
					/ (2 * pi * epsilon * d)
					* normalize((position + atoms[i]->offset) - (molecule->position + molecule->atoms[j]->offset));
			}
			aForce = aForce - aVelocity * rho;
			force = force + aForce;
			m = m + cross(atoms[i]->offset, aForce);
		}
		//printf("force : %f, %f, %f\n", force.x, force.y, force.z);
		//printf("M : %f, %f, %f\n", m.x, m.y, m.z);
	}

	void update() {
		velocity = velocity + force / (mass * hMass) * dt;
		vec3 delta = velocity * dt;
		moveBy(vec2(delta.x, delta.y));

		omega = omega + m / (theta * hMass) * dt;
		float deltaPhi = (omega * dt).z;
		rotateBy(deltaPhi);

	}

	void moveBy(vec2 t) {
		position = position + t;
		for (int i = 0; i < vEdges.size(); i++) {
			vEdges[i] = vEdges[i] + t;
		}
		for (int i = 0; i < atoms.size(); i++) {
			atoms[i]->move(t);
		}
	}

	void rotateBy(float phi) {
		this->phi += phi;
		for (int i = 0; i < vEdges.size(); i++) {
			vEdges[i] = rotate(vEdges[i], position, phi);
		}
		for (int i = 0; i < size; i++) {
			atoms[i]->rotateBy(phi, position);
		}
	}
};

Molecule* m1 = new Molecule();
Molecule* m2 = new Molecule();

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(1.0f);
	m1->create();
	m2->create();
	m1->moveBy(vec2((float)(rand() % 1000) / 1000 - 0.5, (float)(rand() % 1000) / 1000 - 0.5));
	m2->moveBy(vec2((float)(rand() % 1000) / 1000 - 0.5, (float)(rand() % 1000) / 1000 - 0.5));

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.4f, 0.4f, 0.4f, 0);     // background color <- should be in init()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	m1->draw();
	m2->draw();

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
			delete m2;
			m1 = new Molecule();
			m1->create();
			m1->moveBy(vec2((float)(rand() % 1000) / 1000 -0.5, (float)(rand() % 1000) / 1000 - 0.5));
			m2 = new Molecule();
			m2->create();
			m2->moveBy(vec2((float)(rand() % 1000) / 1000 - 0.5, (float)(rand() % 1000) / 1000 - 0.5));
		}break;
		case 'p': m1->moveBy(vec2(0.01, 0)); break;
		case 'o': m1->moveBy(vec2(-0.01, 0)); break;
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
float lastUpdate = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;

	float timeSinceUpdate = sec - lastUpdate;
	for (float t = 0; t < timeSinceUpdate; t += dt) {
		m1->applyForce(m2);
		m2->applyForce(m1);
		m1->update();
		m2->update();
		lastUpdate += dt;
	}
	glutPostRedisplay();

}

vec2 rotate(vec2 point, vec2 pivot, float phi){
	float x = pivot.x + (point.x - pivot.x) * cosf(phi) - (point.y - pivot.y) * sinf(phi);
	float y = pivot.y + (point.x - pivot.x) * sinf(phi) + (point.y - pivot.y) * cosf(phi);

	return vec2(x, y);
}

float distance(vec2 p1, vec2 p2) {
	float x = p1.x - p2.x;
	float y = p1.y - p2.y;

	return sqrt(x * x + y * y);
}