#include "neat_imgui.h"
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"
#include "imgui/imgui_sdl2_setup.h"

static void ImGui_ColoredRect(const ImVec2& size, const ImVec4& color)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    ImGui::RenderFrame(bb.Min, bb.Max, ImGui::ColorConvertFloat4ToU32(color), false, 0);
}

void ImGui_NeatGene(const Gene& gene, bool disabled)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImVec2 size(50, 45);
    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    f64 blend = (clamp(gene.weight, -1.0, 1.0) + 1.0) * 0.5;
    ImVec4 color(1.0 - blend, blend, 0.2, 1.0);
    if(disabled) {
        color = ImVec4(0.4, 0.4, 0.4, 1.0);
    }

    u32 bgColor = ImGui::ColorConvertFloat4ToU32(color);
    ImGui::RenderFrame(pos, pos + size, bgColor, false, 0.0);
    ImGui::RenderFrame(pos, pos + ImVec2(50, 15), 0x80000000, false, 0.0);

    char inNumStr[10];
    char linkStr[32];
    char weightStr[32];

    sprintf(inNumStr, "#%d", gene.historicalMarker);
    sprintf(linkStr, "%d > %d", gene.nodeIn, gene.nodeOut);
    sprintf(weightStr, "%.3f", gene.weight);

    pos.x += 2;
    ImGui::RenderText(pos, inNumStr);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,0,0,1));
    ImGui::RenderText(pos + ImVec2(0, 15), linkStr);
    ImGui::RenderText(pos + ImVec2(0, 30), weightStr);
    ImGui::PopStyleColor(1);
}

void ImGui_NeatGeneList(const Genome* genome)
{
    ImGui::BeginChild((ImGuiID)(intptr_t)genome, ImVec2(300, 140));

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1,1));
    const i32 geneCount = genome->geneCount;
    const i32 perLine = 5;
    for(i32 i = 0; i < geneCount; ++i) {
        ImGui_NeatGene(genome->genes[i], genome->geneDisabled[i]);
        if(((i+1) % perLine) != 0 && i != geneCount-1) {
            ImGui::SameLine();
        }
    }
    ImGui::PopStyleVar(1);

    ImGui::EndChild();
}

void ImGui_NeatNN(const Genome* genome)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const Genome& g = *genome;
    const ImVec2 nodeSpace(30, 30);
    const f32 linkSpaceWidth = 30.f;
    const f32 nodeRadius = 12.f;

    i32 maxLayer = 0;
    i32 maxVPos = 0;
    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        maxVPos = max(maxVPos, g.nodePos[i].vpos);
        maxLayer = max(maxLayer, g.nodePos[i].layer);
    }
    maxVPos++;

    ImVec2 frameSize(maxLayer * (linkSpaceWidth + nodeSpace.x), nodeSpace.y * maxVPos);
    ImVec2 pos = window->DC.CursorPos;
    ImRect bb(pos, pos + frameSize);
    ImGui::ItemSize(bb);

    // compute each node position
    ImVec2 nodePos[128];
    assert(g.totalNodeCount <= 128);

    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        nodePos[i] = pos + ImVec2(nodeSpace.x * 0.5 + (linkSpaceWidth + nodeSpace.x) * g.nodePos[i].layer,
                                  nodeSpace.y * g.nodePos[i].vpos + nodeSpace.y * 0.5f);
    }

    // draw lines
    for(i32 i = 0; i < g.geneCount; ++i) {
        if(g.geneDisabled[i]) continue;
        f64 blend = (clamp(g.genes[i].weight, -1.0, 1.0) + 1.0) * 0.5;
        ImVec4 color4(1.0 - blend, blend, 0.2, 1.0);
        u32 lineCol = ImGui::ColorConvertFloat4ToU32(color4);
        draw_list->AddLine(nodePos[g.genes[i].nodeIn], nodePos[g.genes[i].nodeOut], lineCol, 2.0);
    }

    // draw nodes
    char numStr[5];
    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        draw_list->AddCircleFilled(nodePos[i], nodeRadius, 0xffffffffff, 32);
        sprintf(numStr, "%d", i);
        ImVec2 labelSize = ImGui::CalcTextSize(numStr);
        draw_list->AddText(nodePos[i] + ImVec2(-labelSize.x * 0.5, -labelSize.y * 0.5), 0xFF000000, numStr);
    }
}
