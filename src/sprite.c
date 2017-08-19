#include "sprite.h"
#include "stb_image.h"
#include <stdlib.h>
#include <gl3w.h>
#include <math.h>

typedef struct
{
    f32 md[16];
} Mat4;

inline Mat4 Mat4_mul(const Mat4* m1, const Mat4* m2)
{
    Mat4 m;
    f32* md = m.md;
    const f32* md1 = m1->md;
    const f32* md2 = m2->md;

    md[0] = md1[0] * md2[0] + md1[4] * md2[1] + md1[8] * md2[2] + md1[12] * md2[3];
    md[1] = md1[1] * md2[0] + md1[5] * md2[1] + md1[9] * md2[2] + md1[13] * md2[3];
    md[2] = md1[2] * md2[0] + md1[6] * md2[1] + md1[10] * md2[2] + md1[14] * md2[3];
    md[3] = md1[3] * md2[0] + md1[7] * md2[1] + md1[11] * md2[2] + md1[15] * md2[3];

    md[4] = md1[0] * md2[4] + md1[4] * md2[5] + md1[8] * md2[6] + md1[12] * md2[7];
    md[5] = md1[1] * md2[4] + md1[5] * md2[5] + md1[9] * md2[6] + md1[13] * md2[7];
    md[6] = md1[2] * md2[4] + md1[6] * md2[5] + md1[10] * md2[6] + md1[14] * md2[7];
    md[7] = md1[3] * md2[4] + md1[7] * md2[5] + md1[11] * md2[6] + md1[15] * md2[7];

    md[8] = md1[0] * md2[8] + md1[4] * md2[9] + md1[8] * md2[10] + md1[12] * md2[11];
    md[9] = md1[1] * md2[8] + md1[5] * md2[9] + md1[9] * md2[10] + md1[13] * md2[11];
    md[10] = md1[2] * md2[8] + md1[6] * md2[9] + md1[10] * md2[10] + md1[14] * md2[11];
    md[11] = md1[3] * md2[8] + md1[7] * md2[9] + md1[11] * md2[10] + md1[15] * md2[11];

    md[12] = md1[0] * md2[12] + md1[4] * md2[13] + md1[8] * md2[14] + md1[12] * md2[15];
    md[13] = md1[1] * md2[12] + md1[5] * md2[13] + md1[9] * md2[14] + md1[13] * md2[15];
    md[14] = md1[2] * md2[12] + md1[6] * md2[13] + md1[10] * md2[14] + md1[14] * md2[15];
    md[15] = md1[3] * md2[12] + md1[7] * md2[13] + md1[11] * md2[14] + md1[15] * md2[15];

    return m;
}

inline Mat4 mat4Translate(const Vec2* v2)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = 1.f;
    m.md[5] = 1.f;
    m.md[10] = 1.f;
    m.md[15] = 1.f;

    m.md[12] = v2->x;
    m.md[13] = v2->y;
    return m;
}

inline Mat4 mat4Scale(const Vec2* v2)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = v2->x;
    m.md[5] = v2->y;
    m.md[10] = 1.f;
    m.md[15] = 1.f;
    return m;
}

inline Mat4 mat4Rotate(f32 angle)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = cosf(angle);
    m.md[1] = -sinf(angle);
    m.md[4] = sinf(angle);
    m.md[5] = cosf(angle);
    m.md[10] = 1.f;
    m.md[15] = 1.f;
    return m;
}

#define BATCH_MAX_COUNT (1024)

struct
{
    Mat4 tfMats[BATCH_MAX_COUNT];
    Color3 colors[BATCH_MAX_COUNT];

    GLuint vboQuad;
    GLuint vboModelMats; // model matrix
    GLuint vboColors;
    GLuint vaoSprites;

    GLuint shaderProgram;
    GLint uViewMatrix;
    GLint uTexture;
} state;

enum {
    LAYOUT_POSITION = 0,
    LAYOUT_UV = 1,
    LAYOUT_COLOR = 2,
    LAYOUT_MODEL = 3,
};

GLuint glMakeShader(GLenum type, const char* pFileBuff, i32 fileSize)
{
    GLuint shader = glCreateShader(type);

    // compile
    glShaderSource(shader, 1, &pFileBuff, &fileSize);
    glCompileShader(shader);

    // check result
    GLint compileResult = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileResult);

    if(!compileResult) {
        int logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        char* logBuff = (char*)malloc(logLength);
        glGetShaderInfoLog(shader, logLength, NULL, logBuff);
        LOG("Error [shader compilation]: %s", logBuff);
        free(logBuff);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

i32 initSpriteState(i32 winWidth, i32 winHeight)
{
    glGenBuffers(1, &state.vboQuad);
    glGenBuffers(1, &state.vboModelMats);
    glGenBuffers(1, &state.vboColors);
    glGenVertexArrays(1, &state.vaoSprites);

    glBindVertexArray(state.vaoSprites);

    const f32 quad[] = {
        0.f, 0.f,   0.f, 0.f,
        0.f, 1.f,   0.f, 1.f,
        1.f, 1.f,   1.f, 1.f,

        0.f, 0.f,   0.f, 0.f,
        1.f, 1.f,   1.f, 1.f,
        1.f, 0.f,   1.f, 0.f,
    };

    glBindBuffer(GL_ARRAY_BUFFER, state.vboQuad);
    glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(f32), quad, GL_STATIC_DRAW);

    // vertex position
    glVertexAttribPointer(
        LAYOUT_POSITION,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(f32)*4,
        (void*)0
    );
    glEnableVertexAttribArray(LAYOUT_POSITION);

    // vertex texture uv
    glVertexAttribPointer(
        LAYOUT_UV,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(f32)*4,
        (void*)(sizeof(f32)*2)
    );
    glEnableVertexAttribArray(LAYOUT_UV);

    glBindBuffer(GL_ARRAY_BUFFER, state.vboColors);
    glBufferData(GL_ARRAY_BUFFER, BATCH_MAX_COUNT * sizeof(Color3), 0, GL_DYNAMIC_DRAW);

    // sprite color
    glVertexAttribPointer(
        LAYOUT_COLOR,
        3,
        GL_UNSIGNED_BYTE,
        GL_TRUE,
        sizeof(u8)*3,
        (void*)0
    );
    glEnableVertexAttribArray(LAYOUT_COLOR);
    glVertexAttribDivisor(LAYOUT_COLOR, 1);

    glBindBuffer(GL_ARRAY_BUFFER, state.vboModelMats);
    glBufferData(GL_ARRAY_BUFFER, BATCH_MAX_COUNT * sizeof(Mat4), 0, GL_DYNAMIC_DRAW);

    // sprite model matrix
    for(i32 i = 0; i < 4; ++i) {
        glVertexAttribPointer(
            LAYOUT_MODEL + i,
            4,
            GL_FLOAT,
            GL_FALSE,
            sizeof(Mat4),
            (void*)(sizeof(f32)*4*i));
        glEnableVertexAttribArray(LAYOUT_MODEL + i);
        glVertexAttribDivisor(LAYOUT_MODEL + i, 1);
    }

    const char* vertexShaderStr = MAKE_STR(
          #version 330 core\n
          layout(location = 0) in vec2 position;\n
          layout(location = 1) in vec2 uv;\n
          layout(location = 2) in vec3 color;\n
          layout(location = 3) in mat4 model;\n
          uniform mat4 uViewMatrix;\n

          out vec2 vert_uv;
          out vec3 vert_color;

          void main()\n
          {\n
              vert_uv = uv;
              vert_color = color;
              gl_Position = uViewMatrix * model * vec4(position, 0.0, 1.0);\n
          }
    );

    const char* fragmentShaderStr = MAKE_STR(
        #version 330 core\n
        uniform sampler2D uTexture;

        in vec2 vert_uv;\n
        in vec3 vert_color;\n
        out vec4 fragmentColor;\n

        void main()\n
        {\n
            vec4 tex = texture(uTexture, vert_uv);
            fragmentColor = tex * vec4(vert_color, 1.0);\n
        }
    );

    GLuint vertexShader = glMakeShader(GL_VERTEX_SHADER, vertexShaderStr, strlen(vertexShaderStr));
    GLuint fragmentShader = glMakeShader(GL_FRAGMENT_SHADER, fragmentShaderStr, strlen(fragmentShaderStr));

    if(!vertexShader | !fragmentShader) {
        return FALSE;
    }

    state.shaderProgram = glCreateProgram();
    glAttachShader(state.shaderProgram, vertexShader);
    glAttachShader(state.shaderProgram, fragmentShader);
    glLinkProgram(state.shaderProgram);

    // check
    GLint linkResult = GL_FALSE;
    glGetProgramiv(state.shaderProgram, GL_LINK_STATUS, &linkResult);

    if(!linkResult) {
        int logLength = 0;
        glGetProgramiv(state.shaderProgram, GL_INFO_LOG_LENGTH, &logLength);
        char* logBuff =  (char*)malloc(logLength);
        glGetProgramInfoLog(state.shaderProgram, logLength, NULL, logBuff);
        LOG("Error [program link]: %s", logBuff);
        free(logBuff);
        glDeleteProgram(state.shaderProgram);
        return FALSE;
    }

    state.uViewMatrix = glGetUniformLocation(state.shaderProgram, "uViewMatrix");
    state.uTexture = glGetUniformLocation(state.shaderProgram, "uTexture");

    i32 left = 0;
    i32 right = winWidth;
    i32 top = 0;
    i32 bottom = winHeight;
    f32 nearPlane = -10.f;
    f32 farPlane = 10.f;

    Mat4 ortho;
    memset(&ortho, 0, sizeof(ortho));
    ortho.md[0] = 1.f;
    ortho.md[5] = 1.f;
    ortho.md[10] = 1.f;
    ortho.md[15] = 1.f;

    ortho.md[0] = 2.f / (right - left);
    ortho.md[5] = 2.f / (top - bottom);
    ortho.md[10] = -2.f / (farPlane - nearPlane);
    ortho.md[12] = -((right + left) / (right - left));
    ortho.md[13] = -((top + bottom) / (top - bottom));
    ortho.md[14] = -((farPlane + nearPlane) / (farPlane - nearPlane));

    glUseProgram(state.shaderProgram);
    glUniformMatrix4fv(state.uViewMatrix, 1, GL_FALSE, ortho.md);

    // culling
    //glEnable(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);
    //glCullFace(GL_BACK);

    // alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glViewport(0, 0, winWidth, winHeight);

    return TRUE;
}

i32 loadTexture(const char* path)
{
    i32 width, height, comp;
    u8* data = stbi_load(path, &width, &height, &comp, 4);
    if(!data) {
        LOG("ERROR: could not load (%s) reason: %s", path, stbi_failure_reason());
        return -1;
    }

    i32 texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 width, height,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 data);

    stbi_image_free(data);
    LOG("texture loaded (%s) w=%d h=%d c=%d", path, width, height, comp);

    return texture;
}

void drawSpriteBatch(i32 textureId, const Transform* transform, const Color3* color, const i32 count)
{
    for(i32 i = 0; i < count; ++i) {
        Mat4 tr = mat4Translate(&transform[i].pos);
        Vec2 center = {-transform[i].center.x, -transform[i].center.y};
        Mat4 cent = mat4Translate(&center);
        Mat4 rot = mat4Rotate(-transform[i].rot);
        Mat4 sc = mat4Scale(&transform[i].size);
        Mat4 trrot = Mat4_mul(&tr, &rot);
        trrot = Mat4_mul(&trrot, &cent);
        state.tfMats[i] = Mat4_mul(&trrot, &sc);
    }

    glBindBuffer(GL_ARRAY_BUFFER, state.vboModelMats);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(Mat4), state.tfMats);

    glBindBuffer(GL_ARRAY_BUFFER, state.vboColors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(Color3), color);

    glBindVertexArray(state.vaoSprites);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureId);

    glUseProgram(state.shaderProgram);
    glUniform1i(state.uTexture, 1);

    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
}
