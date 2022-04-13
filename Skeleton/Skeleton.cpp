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

const char* const vertexSource = R"(
	#version 330
	precision highp float;
	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;
	void main() {
		vec4 v = vec4(vp.x, vp.y, 0, 1) * MVP;
		float w = sqrt(pow(v.x, 2) + pow(v.y, 2) + 1.0f);
		gl_Position = vec4(v.x / (w + 1), v.y / (w + 1), 0, 1);
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 outColor;
	void main() {
		outColor = vec4(color, 1);
	}
)";

vec2 rotate(vec2 point, vec2 pivot, float phi);
float distance(vec2 p1, vec2 p2);


int maxMass = 50;
int maxCharge = 100;
float eCharge = 1.602176634f * pow(10.0f, -1.0f);	
float hMass = 1.66f *pow(10.0f, -2.0f);				
float epsilon = 8.854187817f;					
float Dt = 0.01f;
float dt = 0.01f;
float rho = 1.5f;
float wX = 2.0f;
float wY = 2.0f;

// 2D camera - source: camera class from https://online.vik.bme.hu/mod/url/view.php?id=18342 - Színátmenetes háromszög és töröttvonal
class Camera2D {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera2D() : wCenter(0, 0), wSize(wX, wY) { }

	mat4 V() { return TranslateMatrix(-wCenter); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }

	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;
GPUProgram gpuProgram;


class Atom {
public:
	int ID = 0;
	int mass;
	int charge;
	int bonds;
	std::vector<Atom*> bondedWith;
	vec2 offset = NULL;
	bool assigned = false;

	unsigned int vao = 0;
	unsigned int vbo = 0;
	const int nv = 20;
	float phi = 0;

	float sx = wX/2;
	float sy = wY/2;
	std::vector<vec2> vertices;
	float r = 0.15;

	Atom() {
		mass = rand() % maxMass + 1;
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
			float fi = i * 2 * M_PI / nv;
			float tr = r * (fabs(cbrt(mass) / cbrt((float)maxMass)));
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

	void draw() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, nv * sizeof(vec2), &vertices[0]);
		mat4 MVPTransform = camera.V();
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
	int size;
	float mass = 0;
	float theta = 0;					

	std::vector<Atom*> atoms;
	std::vector<vec2> edges;
	std::vector<vec2> vEdges;
	int nvEdge = 20;
	vec2 position = vec2(0, 0);
	float phi = 0;

	vec3 velocity = (0.0f, 0.0f, 0.0f);
	vec3 omega = (0.0f, 0.0f, 0.0f);
	vec3 force = (0.0f, 0.0f, 0.0f);
	vec3 m = (0.0f, 0.0f, 0.0f);

	vec2 center;
	float sx = wX/2;
	float sy = wY/2;
	float timeOfLastUpdate = 0;
	unsigned int vao;
	unsigned int vbo;
public:
	Molecule() {
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
			mass += atoms[i]->mass;
			int delta = rand() % maxCharge + 1;
			atoms[i]->charge -= delta;
			atoms[(i + 1) % size]->charge += delta;
		}

		int rootIndex = 0;		
		int maxBonds = 0;
		for (int i = 0; i < size; i++) {
			if (atoms[i]->bonds > maxBonds) {
				maxBonds = atoms[i]->bonds;
				rootIndex = i;
			}
		}

		atoms[rootIndex]->offset = vec2(0, 0);
		atoms[rootIndex]->assigned = true;
		build(*atoms[rootIndex]);

		for (int i = 0; i < size; i++) {
			center = center + atoms[i]->offset * atoms[i]->mass;
		}
		center = center / mass;

		for (int i = 0; i < size; i++) {
			atoms[i]->offset = atoms[i]->offset - center;
		}

		for (int i = 0; i < edges.size(); i++) {
			edges[i] = edges[i] - center;
		}

		for (int i = 0; i < atoms.size(); i++) {
			float r = length(atoms[i]->offset);
			theta += atoms[i]->mass * r * r;
		}

		for (int i = 0; i < edges.size(); i+=2) {
			for (int j = 0; j < nvEdge; j++) {
				float t = (float)j / (nvEdge-1);
				vEdges.push_back(vec2(edges[i].x * t + edges[i+1].x * (1.0f - t), edges[i].y * t + edges[i + 1].y * (1.0f - t)));
			}
		}
	}

	void build(Atom root) {
		for (int i = 0; i < root.bonds; i++) {
			if (!root.bondedWith[i]->assigned) {
				float angle = M_PI / 180 * (rand() % 360);
				float l = (0.5f + (float)(rand() % 50) * 0.02f);
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

	void draw() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, vEdges.size() * sizeof(vec2), &vEdges[0]);

		mat4 MVPTransform = camera.V();
		gpuProgram.setUniform(MVPTransform, "MVP");
		gpuProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");

		glBindVertexArray(vao);
		for (int i = 0; i < edges.size()/2; i++) {
			glDrawArrays(GL_LINE_STRIP, i*nvEdge, nvEdge);
		}

		for (int i = 0; i < size; i++) {
			atoms[i]->draw();
		}
	}

	void applyForce(Molecule* molecule) {
		force = vec3(0, 0, 0);
		m = vec3(0, 0, 0);
		for (int i = 0; i < size; i++) {
			vec3 aVelocity = velocity + cross(omega, atoms[i]->offset);

			vec3 aForce = vec3(0, 0, 0);
			for (int j = 0; j < molecule->size; j++) {
				float d = max(distance(position + atoms[i]->offset, molecule->position + molecule->atoms[j]->offset), 0.05);
				aForce = aForce + (atoms[i]->charge * molecule->atoms[j]->charge * pow(eCharge, 2))
					/ (2 * M_PI * epsilon * d)
					* normalize((position + atoms[i]->offset) - (molecule->position + molecule->atoms[j]->offset));
			}
			aForce = aForce - aVelocity * rho;
			force = force + aForce;
			m = m + cross(atoms[i]->offset, aForce);
		}
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

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(1.0f);
	m1->create();
	m2->create();
	m1->moveBy(vec2((float)(rand() % 1000) / 500 - 1.0, (float)(rand() % 1000) / 500 - 1.0));
	m2->moveBy(vec2((float)(rand() % 1000) / 500 - 1.0, (float)(rand() % 1000) / 500 - 1.0));

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0.4f, 0.4f, 0.4f, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m1->draw();
	m2->draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
		case 's': camera.Pan(vec2(-0.1, 0)); break;
		case 'd': camera.Pan(vec2(+0.1, 0)); break;
		case 'e': camera.Pan(vec2(0, 0.1)); break;
		case 'x': camera.Pan(vec2(0, -0.1)); break;
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
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

float lastUpdate = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 1000.0f;

	float timeSinceUpdate = sec - lastUpdate;
	dt = min(timeSinceUpdate, Dt);
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