#include "sprite.h"
#include "stb_image.h"
#include <stdlib.h>
#include <gl3w.h>

#define BATCH_MAX_COUNT (1024)

struct
{
    Mat4 tfMats[BATCH_MAX_COUNT];
    Color3 colors[BATCH_MAX_COUNT];

    GLuint vboQuad;
    GLuint vboModelMats; // model matrix
    GLuint vboColors;
    GLuint vaoSprites;

    GLuint vboLines;
    GLuint vaoLines;

    GLuint programSprite;
    GLint uViewMatrix;
    GLint uTexture;

    struct {
        GLuint program;
        GLint uViewMatrix;
    } shaderLine;
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

i32 initSpriteShader()
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

    state.programSprite = glCreateProgram();
    glAttachShader(state.programSprite, vertexShader);
    glAttachShader(state.programSprite, fragmentShader);
    glLinkProgram(state.programSprite);

    // check
    GLint linkResult = GL_FALSE;
    glGetProgramiv(state.programSprite, GL_LINK_STATUS, &linkResult);

    if(!linkResult) {
        int logLength = 0;
        glGetProgramiv(state.programSprite, GL_INFO_LOG_LENGTH, &logLength);
        char* logBuff =  (char*)malloc(logLength);
        glGetProgramInfoLog(state.programSprite, logLength, NULL, logBuff);
        LOG("Error [program link]: %s", logBuff);
        free(logBuff);
        glDeleteProgram(state.programSprite);
        return FALSE;
    }

    state.uViewMatrix = glGetUniformLocation(state.programSprite, "uViewMatrix");
    state.uTexture = glGetUniformLocation(state.programSprite, "uTexture");

    if(state.uViewMatrix == -1 || state.uTexture == -1) {
        return FALSE;
    }

    return TRUE;
}

i32 initLineShader()
{
    // LINES
    glGenBuffers(1, &state.vboLines);
    glGenVertexArrays(1, &state.vaoLines);

    glBindVertexArray(state.vaoLines);

    glBindBuffer(GL_ARRAY_BUFFER, state.vboLines);
    glBufferData(GL_ARRAY_BUFFER, BATCH_MAX_COUNT * sizeof(Line), 0, GL_DYNAMIC_DRAW);

    // vertex position
    glVertexAttribPointer(
        LAYOUT_POSITION,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vec2)+sizeof(Color4),
        (void*)0
    );
    glEnableVertexAttribArray(LAYOUT_POSITION);

    // line color
    glVertexAttribPointer(
        LAYOUT_COLOR,
        4,
        GL_UNSIGNED_BYTE,
        GL_TRUE,
        sizeof(Vec2)+sizeof(Color4),
        (void*)(sizeof(Vec2))
    );
    glEnableVertexAttribArray(LAYOUT_COLOR);

    const char* vertexShaderStr = MAKE_STR(
          #version 330 core\n
          layout(location = 0) in vec2 position;\n
          layout(location = 2) in vec4 color;\n
          uniform mat4 uViewMatrix;\n

          out vec4 vert_color;

          void main()\n
          {\n
              vert_color = color;
              gl_Position = uViewMatrix * vec4(position, 0.0, 1.0);\n
          }
    );

    const char* fragmentShaderStr = MAKE_STR(
        #version 330 core\n

        in vec4 vert_color;\n
        out vec4 fragmentColor;\n

        void main()\n
        {\n
            fragmentColor = vert_color;\n
        }
    );

    GLuint vertexShader = glMakeShader(GL_VERTEX_SHADER, vertexShaderStr, strlen(vertexShaderStr));
    GLuint fragmentShader = glMakeShader(GL_FRAGMENT_SHADER, fragmentShaderStr, strlen(fragmentShaderStr));

    if(!vertexShader | !fragmentShader) {
        return FALSE;
    }

    state.shaderLine.program = glCreateProgram();
    glAttachShader(state.shaderLine.program, vertexShader);
    glAttachShader(state.shaderLine.program, fragmentShader);
    glLinkProgram(state.shaderLine.program);

    // check
    GLint linkResult = GL_FALSE;
    glGetProgramiv(state.shaderLine.program, GL_LINK_STATUS, &linkResult);

    if(!linkResult) {
        int logLength = 0;
        glGetProgramiv(state.shaderLine.program, GL_INFO_LOG_LENGTH, &logLength);
        char* logBuff =  (char*)malloc(logLength);
        glGetProgramInfoLog(state.shaderLine.program, logLength, NULL, logBuff);
        LOG("Error [program link]: %s", logBuff);
        free(logBuff);
        glDeleteProgram(state.shaderLine.program);
        return FALSE;
    }

    state.shaderLine.uViewMatrix = glGetUniformLocation(state.shaderLine.program, "uViewMatrix");

    if(state.shaderLine.uViewMatrix == -1) {
        return FALSE;
    }

    return TRUE;
}

i32 initSpriteState(i32 winWidth, i32 winHeight)
{
    if(!initSpriteShader()) {
        LOG("ERROR: could init sprite shader correctly");
        return FALSE;
    }

    if(!initLineShader()) {
        LOG("ERROR: could init line shader correctly");
        return FALSE;
    }

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

void setView(i32 x, i32 y, i32 width, i32 height)
{
    Mat4 ortho = mat4Ortho(0, width, 0, height, -10.f, 10.f);
    Vec2 v = {-x, -y};
    Mat4 tr = mat4Translate(&v);
    Mat4 view = mat4Mul(&ortho, &tr);

    glUseProgram(state.programSprite);
    glUniformMatrix4fv(state.uViewMatrix, 1, GL_FALSE, view.md);

    glUseProgram(state.shaderLine.program);
    glUniformMatrix4fv(state.shaderLine.uViewMatrix, 1, GL_FALSE, view.md);
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
        Mat4 trrot = mat4Mul(&tr, &rot);
        trrot = mat4Mul(&trrot, &cent);
        state.tfMats[i] = mat4Mul(&trrot, &sc);
    }

    glBindBuffer(GL_ARRAY_BUFFER, state.vboModelMats);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(Mat4), state.tfMats);

    glBindBuffer(GL_ARRAY_BUFFER, state.vboColors);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(Color3), color);

    glBindVertexArray(state.vaoSprites);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureId);

    glUseProgram(state.programSprite);
    glUniform1i(state.uTexture, 1);

    glDrawArraysInstanced(GL_TRIANGLES, 0, 6, count);
}

void drawLineBatch(const Line* lines, const i32 count)
{
    glBindBuffer(GL_ARRAY_BUFFER, state.vboLines);
    glBufferSubData(GL_ARRAY_BUFFER, 0, count * sizeof(Line), lines);

    glBindVertexArray(state.vaoLines);

    glUseProgram(state.shaderLine.program);

    glDrawArrays(GL_LINES, 0, count * 2);
}
