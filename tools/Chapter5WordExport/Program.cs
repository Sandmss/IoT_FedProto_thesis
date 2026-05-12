using System.IO;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using A = DocumentFormat.OpenXml.Drawing;
using DW = DocumentFormat.OpenXml.Drawing.Wordprocessing;
using PIC = DocumentFormat.OpenXml.Drawing.Pictures;

var outputPath = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.Combine(Directory.GetCurrentDirectory(), "第五章规范版.docx");

var root = @"E:\IoT_FedProto";
var assetDir = Path.Combine(root, ".tmp", "chapter5_word_assets");

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
if (File.Exists(outputPath))
{
    File.Delete(outputPath);
}

using var doc = WordprocessingDocument.Create(outputPath, WordprocessingDocumentType.Document);
var mainPart = doc.AddMainDocumentPart();
mainPart.Document = new Document(new Body());

ConfigureStyles(mainPart);

var body = mainPart.Document.Body!;

AppendChapterTitle(body, "第5章 实验设计与结果分析");

AppendSectionTitle(body, "5.1 实验设置");
AppendBodyParagraph(body, "本文实验在 Python 3.10 环境下完成，主要依赖 PyTorch 2.5.1、NumPy 2.0.1、scikit-learn 1.7.2、pandas 2.3.3 和 matplotlib 3.10.8。为保证不同方法之间具有可比性，各组实验采用统一的数据划分方式和评价指标设置。主实验参数包括：客户端数量 20，输入特征维度 77，原型特征维度 512，本地批大小 10，本地训练轮数 1，全局轮数 1000，早停耐心值 100。对比算法包括 Local、FedAvg 和 FedProto；同构模型包括 MLP、CNN1D 和 Transformer1D，异构模型采用 MLP 与 CNN1D 的混合配置。上述实验设置与前文关于 Non-IID 客户端划分和原型级通信机制的分析保持一致，可为后续性能比较与轻量化分析提供统一条件。");

AppendSectionTitle(body, "5.2 数据集与客户端划分说明");
AppendBodyParagraph(body, "本文实验数据说明以实际预处理流程和数据组织结果为准。原始流量文件经统一导入后，依次执行数据清洗、特征编码和样本划分。预处理流程主要包括删除标识性字段、将特征列转换为数值类型、去除空值与重复样本、进行标签编码，以及采用最小-最大归一化方法完成数值缩放处理。该处理方式与恶意流量检测中常见的统计特征建模思路保持一致，有助于在保留主要区分信息的同时降低后续模型训练难度[10][20]。");
AppendBodyParagraph(body, "根据数据组织结果，输入样本已整理为 77 维特征向量；训练任务中类别数设置为 15，因此本文实验按 15 类任务进行配置，编码后的标签范围记为 0 至 14。为保证实验口径一致，类别名称与编码映射关系由预处理阶段统一维护，正文主要给出类别规模、样本数量和划分方式等与实验比较直接相关的信息。");
AppendBodyParagraph(body, "客户端划分采用按类别分配的 Non-IID 方式。本实验共设置 20 个客户端，每个客户端均具有独立的训练集与测试集。根据现有划分结果，每个客户端包含 5 个类别，每个客户端训练集包含 375 条样本，测试集包含 125 条样本，全局总计为 7500 条训练样本和 2500 条测试样本。该结果与“每个客户端仅保留部分类别，并按 0.75:0.25 比例划分训练集和测试集”的预处理原则一致。");
AppendBodyParagraph(body, "上述划分方式能够较好模拟物联网场景中的设备差异。原因在于不同终端在实际运行中并不会长期接触完全一致的流量类别集合，部分节点可能只观察到局部攻击模式，另一些节点则以正常业务流量为主。将每个客户端限制为部分类别并保留独立的训练集和测试集，有助于刻画“类别覆盖不完整、样本规模有限、节点分布不同”的实际协同环境，这也更接近联邦异常检测研究中常见的 Non-IID 设定[7][14]。");

AppendSectionTitle(body, "5.3 主实验结果对比");
AppendBodyParagraph(body, "表5.1和表5.2分别给出了不同模型在 Local、FedAvg 和 FedProto 条件下的检测性能结果与轻量化开销结果。");

AppendTableTitle(body, "表5.1 不同模型与联邦机制下的检测性能比较");
AppendThreeLineTable(
    body,
    new[] { "模型", "算法", "轮数", "准确率", "宏AUC", "微AUC", "F1值", "漏报率", "误报率" },
    new[]
    {
        new[] { "MLP", "Local", "887", "0.9392", "0.9917", "0.9923", "0.9384", "0.0000", "0.1480" },
        new[] { "MLP", "FedAvg", "1001", "0.8684", "0.9919", "0.9916", "0.5557", "0.0000", "0.0000" },
        new[] { "MLP", "FedProto", "893", "0.9384", "0.9918", "0.9924", "0.9377", "0.0000", "0.1480" },
        new[] { "CNN1D", "Local", "690", "0.9416", "0.9915", "0.9927", "0.9412", "0.0000", "0.1340" },
        new[] { "CNN1D", "FedAvg", "1001", "0.8976", "0.9962", "0.9956", "0.6087", "0.0000", "0.0000" },
        new[] { "CNN1D", "FedProto", "1000", "0.9484", "0.9918", "0.9929", "0.9481", "0.0000", "0.1120" },
        new[] { "Transformer1D", "Local", "896", "0.9292", "0.9857", "0.9871", "0.9280", "0.0000", "0.1480" },
        new[] { "Transformer1D", "FedAvg", "1001", "0.8856", "0.9948", "0.9954", "0.5891", "0.0000", "0.0000" },
        new[] { "Transformer1D", "FedProto", "1000", "0.9300", "0.9888", "0.9902", "0.9288", "0.0105", "0.1380" }
    },
    new[] { 1100, 1000, 650, 900, 900, 900, 800, 1388, 1388 });

AppendTableTitle(body, "表5.2 不同模型与联邦机制下的轻量化开销比较");
AppendThreeLineTable(
    body,
    new[] { "模型", "算法", "平均推理时延（ms/sample）", "每轮通信参数量" },
    new[]
    {
        new[] { "MLP", "Local", "0.0362", "0" },
        new[] { "MLP", "FedAvg", "0.0360", "1828440" },
        new[] { "MLP", "FedProto", "0.0356", "51200" },
        new[] { "CNN1D", "Local", "0.0691", "0" },
        new[] { "CNN1D", "FedAvg", "0.0722", "3220440" },
        new[] { "CNN1D", "FedProto", "0.0808", "51200" },
        new[] { "Transformer1D", "Local", "0.1413", "0" },
        new[] { "Transformer1D", "FedAvg", "0.1500", "3076440" },
        new[] { "Transformer1D", "FedProto", "0.1507", "51200" }
    },
    new[] { 1600, 1400, 3000, 3026 });

AppendBodyParagraph(body, "需要说明的是，表5.1中的准确率、F1值、漏报率和误报率均来自训练全过程日志记录中的单项最优结果，并不一定对应同一轮次。因此，表中数据主要用于反映不同方法在本实验配置下的性能上界与总体趋势，而非某一固定轮次的完整指标切片。为避免引入未经复核的新结果，本文仍按照现有实验记录进行整理，并在后文分析中重点关注不同方法之间的相对差异。");

AppendImage(body, mainPart, Path.Combine(assetDir, "fig5-1-accuracy-f1.png"), 6.0, 2507, 1033);
AppendFigureCaption(body, "图5.1 不同模型与联邦机制下的准确率和 F1 对比图");

AppendBodyParagraph(body, "从实验结果看，FedProto 在三类模型上均保持了较高检测性能。其中，CNN1D + FedProto 取得了 0.9484 的最高准确率和 0.9481 的最高 F1 值，说明在 77 维流量统计特征条件下，一维卷积结构与原型协同机制具有较好的适配性。MLP + FedProto 的准确率与本地训练结果接近，说明原型对齐并未显著削弱轻量模型的本地判别能力。Transformer1D + FedProto 的结果与 Local 基本相当，但推理代价更高，说明更强表达模型在该任务设置下并未带来明显收益。这一结果与前文关于轻量模型适用于资源受限 IoT 终端的分析基本一致，也与通信高效联邦检测研究中的一般趋势相符[14][19]。");
AppendBodyParagraph(body, "FedAvg 在多组实验中出现了 AUC 较高而 F1 明显偏低的现象。以 CNN1D 为例，其宏AUC 达到 0.9962，但 F1 仅为 0.6087。该现象表明模型在排序意义上仍具有较强区分能力，但在最终类别决策阶段存在明显偏斜。在 Non-IID 多分类场景下，一些类别可能长期受到多数客户端分布的影响而被弱化，从而导致少数类预测不足。这说明在联邦检测任务中，仅凭 AUC 指标难以充分反映模型的实际分类效果，更有必要结合 F1、FNR 和 FPR 等指标进行综合判断[7][14]。");

AppendImage(body, mainPart, Path.Combine(assetDir, "fig5-2-cnn1d-tsne-triptych.png"), 6.1, 3190, 832);
AppendFigureCaption(body, "图5.2 CNN1D 在 Local、FedAvg 与 FedProto 条件下的特征分布 t-SNE 可视化结果");

AppendBodyParagraph(body, "由图5.2可见，FedProto 条件下的同类特征分布相对更集中，不同类别之间的分离程度也更清晰，这与其在主实验中的综合指标表现相一致。此外，表中部分方案的漏报率或误报率出现 0。此类结果仅说明在该测试集和当前统计口径下，该轮未观察到相应类型的错误，不应被直接解释为模型在真实部署环境中完全不存在漏报或误报。");

AppendSectionTitle(body, "5.4 轻量化效果分析");
AppendBodyParagraph(body, "本文从通信量和推理时延两个维度分析系统的轻量化效果。对 FedProto 而言，本实验配置下每轮通信参数量固定为 51200，其计算依据为：20 个客户端全部参与训练、每个客户端保留 5 个类别、每个类别上传 1 个 512 维原型，因此每轮上传的原型标量总数为");
AppendCenteredFormula(body, "20×5×512=51200");
AppendBodyParagraph(body, "需要说明的是，这里的 51200 表示每轮上传的浮点标量数量，属于参数量级统计，而非字节数。若按 32 位浮点数计算，仅上传部分约对应 204800 字节，实际网络开销还会受到序列化和协议封装方式影响。");
AppendBodyParagraph(body, "对于 FedAvg，表5.2中的 1828440、3220440 和 3076440 表示每轮上传的模型参数标量总数。其统计口径可理解为“参与客户端数 × 单客户端模型参数量”的量级估计，因此通信开销会随着模型规模增加而上升。相比之下，FedProto 的通信量主要由客户端数量、每客户端类别数和原型维度决定，不再随模型参数规模线性增长。这也正是原型通信机制相较于参数通信机制的重要优势之一[9][19]。");

AppendImage(body, mainPart, Path.Combine(assetDir, "fig5-3-comm-latency.png"), 6.0, 2573, 1078);
AppendFigureCaption(body, "图5.3 不同方法每轮通信量与平均推理时延对比结果");

AppendBodyParagraph(body, "由图5.3可见，FedProto 在通信量控制方面具有稳定优势，而 CNN1D 在检测性能与时延之间表现出较好的平衡。在推理时延方面，MLP 的单样本平均推理时延最低，适用于资源受限终端；CNN1D 时延略高，但检测性能最优；Transformer1D 时延最高，而精度提升并不明显。结合准确率、F1、通信量和推理时延四项指标，CNN1D + FedProto 在本实验中表现出较好的综合效果。");

AppendSectionTitle(body, "5.5 异构客户端实验分析");
AppendBodyParagraph(body, "为验证方法在异构客户端条件下的适用性，本文进一步构建了 MLP 与 CNN1D 的异构 FedProto 实验，结果如表5.3所示。");

AppendTableTitle(body, "表5.3 异构客户端条件下 FedProto 实验结果");
AppendThreeLineTable(
    body,
    new[] { "模型配置", "算法", "轮数", "准确率", "宏AUC", "微AUC", "F1值", "漏报率", "误报率", "平均推理时延（ms/sample）", "每轮通信参数量" },
    new[]
    {
        new[] { "MLP-CNN1D异构组合", "FedProto", "832", "0.9416", "0.9921", "0.9926", "0.9411", "0.0000", "0.1260", "0.0541", "51200" }
    },
    new[] { 1400, 750, 550, 650, 650, 650, 650, 650, 650, 1150, 1276 });

AppendImage(body, mainPart, Path.Combine(assetDir, "fig5-4-heterogeneous-tsne.png"), 5.8, 1333, 1000);
AppendFigureCaption(body, "图5.4 异构 MLP 与 CNN1D 组合在 FedProto 下的特征分布可视化结果");

AppendBodyParagraph(body, "实验结果表明，异构 MLP + CNN1D + FedProto 方案在准确率上达到 0.9416，与同构场景中的较优结果接近。这说明，只要不同客户端模型输出维度保持一致，即使本地模型结构不同，也能够通过共享类别原型实现稳定协同。从系统设计角度看，该结果验证了统一表征结构与统一原型空间设计的可行性，也从实验层面说明了 FedProto 在异构客户端场景下的适用性[9][15]。");

AppendSectionTitle(body, "5.6 结果讨论与局限");
AppendBodyParagraph(body, "本章实验结果表明，在本实验设置下，FedProto 相比 FedAvg 在 Non-IID 条件下表现出更低的通信开销，并保持了较高的准确率和 F1。三类本地模型中，CNN1D 的整体表现相对稳定，说明一维卷积对现有统计特征的局部模式具有较好的建模能力。联邦检测效果不宜仅依据单一高指标进行判断，尤其在 FedAvg 出现 AUC 与 F1 分裂时，更需要结合 FNR、FPR 等指标进行综合分析。");
AppendBodyParagraph(body, "本文仍存在一定局限。实验主要基于离线数据集，尚未在真实边缘设备和在线流量环境中完成部署验证；异构实验目前仅覆盖 MLP 与 CNN1D 的组合，模型异构范围仍较有限。与此同时，本文主要关注原型通信带来的轻量化收益，尚未进一步引入剪枝、量化或蒸馏等更细粒度的压缩策略。虽然原型通信比参数通信更紧凑，但其潜在隐私边界仍有必要进一步分析。");

body.Append(new SectionProperties(
    new PageSize { Width = 11906U, Height = 16838U },
    new PageMargin
    {
        Top = 1134,
        Right = 1134U,
        Bottom = 1134,
        Left = 1134U,
        Header = 851U,
        Footer = 992U,
        Gutter = 0U
    }));

mainPart.Document.Save();
Console.WriteLine(outputPath);

static void ConfigureStyles(MainDocumentPart mainPart)
{
    var stylePart = mainPart.AddNewPart<StyleDefinitionsPart>();
    stylePart.Styles = new Styles(
        new DocDefaults(
            new RunPropertiesDefault(
                new RunPropertiesBaseStyle(
                    new RunFonts
                    {
                        Ascii = "Times New Roman",
                        HighAnsi = "Times New Roman",
                        EastAsia = "SimSun",
                        ComplexScript = "Times New Roman"
                    },
                    new FontSize { Val = "24" },
                    new FontSizeComplexScript { Val = "24" },
                    new Languages { Val = "en-US", EastAsia = "zh-CN" }
                )
            ),
            new ParagraphPropertiesDefault(
                new ParagraphPropertiesBaseStyle(
                    new SpacingBetweenLines
                    {
                        Line = "440",
                        LineRule = LineSpacingRuleValues.Exact
                    }
                )
            )
        ));
}

static void AppendChapterTitle(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "0", After = "220", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            new RunProperties(
                new RunFonts { Ascii = "Times New Roman", HighAnsi = "Times New Roman", EastAsia = "SimHei" },
                new Bold(),
                new BoldComplexScript(),
                new FontSize { Val = "32" },
                new FontSizeComplexScript { Val = "24" }),
            new Text(text)));
    body.Append(p);
}

static void AppendSectionTitle(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Left },
            new SpacingBetweenLines { Before = "160", After = "100", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            new RunProperties(
                new RunFonts { Ascii = "Times New Roman", HighAnsi = "Times New Roman", EastAsia = "SimHei" },
                new Bold(),
                new BoldComplexScript(),
                new FontSize { Val = "28" },
                new FontSizeComplexScript { Val = "24" }),
            new Text(text)));
    body.Append(p);
}

static void AppendBodyParagraph(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Both },
            new Indentation { FirstLineChars = 200, FirstLine = "200" },
            new SpacingBetweenLines { Before = "0", After = "0", Line = "440", LineRule = LineSpacingRuleValues.Exact }),
        new Run(
            new RunProperties(
                new RunFonts
                {
                    Ascii = "Times New Roman",
                    HighAnsi = "Times New Roman",
                    EastAsia = "SimSun",
                    ComplexScript = "Times New Roman"
                },
                new FontSize { Val = "24" },
                new FontSizeComplexScript { Val = "24" }),
            new Text(text) { Space = SpaceProcessingModeValues.Preserve }));
    body.Append(p);
}

static void AppendCenteredFormula(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "80", After = "80", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            new RunProperties(
                new RunFonts
                {
                    Ascii = "Times New Roman",
                    HighAnsi = "Times New Roman",
                    EastAsia = "Times New Roman",
                    ComplexScript = "Times New Roman"
                },
                new FontSize { Val = "24" },
                new FontSizeComplexScript { Val = "24" }),
            new Text(text)));
    body.Append(p);
}

static void AppendTableTitle(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "120", After = "80", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            new RunProperties(
                new RunFonts
                {
                    Ascii = "Times New Roman",
                    HighAnsi = "Times New Roman",
                    EastAsia = "SimSun",
                    ComplexScript = "Times New Roman"
                },
                new FontSize { Val = "21" },
                new FontSizeComplexScript { Val = "21" }),
            new Text(text)));
    body.Append(p);
}

static void AppendFigureCaption(Body body, string text)
{
    var p = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "80", After = "120", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            new RunProperties(
                new RunFonts
                {
                    Ascii = "Times New Roman",
                    HighAnsi = "Times New Roman",
                    EastAsia = "SimSun",
                    ComplexScript = "Times New Roman"
                },
                new FontSize { Val = "21" },
                new FontSizeComplexScript { Val = "21" }),
            new Text(text)));
    body.Append(p);
}

static void AppendThreeLineTable(Body body, string[] headers, string[][] rows, int[] columnWidths)
{
    var table = new Table();
    const int contentWidth = 9638;
    var originalSum = columnWidths.Sum();
    var scaledWidths = columnWidths
        .Select(w => (int)Math.Round(w * (contentWidth / (double)originalSum)))
        .ToArray();
    var delta = contentWidth - scaledWidths.Sum();
    scaledWidths[^1] += delta;

    var tblPr = new TableProperties(
        new TableJustification { Val = TableRowAlignmentValues.Center },
        new TableWidth { Width = contentWidth.ToString(), Type = TableWidthUnitValues.Dxa },
        new TableLayout { Type = TableLayoutValues.Fixed },
        new TableCellMarginDefault(
            new TopMargin { Width = "40", Type = TableWidthUnitValues.Dxa },
            new BottomMargin { Width = "40", Type = TableWidthUnitValues.Dxa },
            new TableCellLeftMargin { Width = 40, Type = TableWidthValues.Dxa },
            new TableCellRightMargin { Width = 40, Type = TableWidthValues.Dxa }),
        new TableBorders(
            new TopBorder { Val = BorderValues.Single, Size = 8, Color = "000000" },
            new BottomBorder { Val = BorderValues.Single, Size = 8, Color = "000000" },
            new LeftBorder { Val = BorderValues.None, Size = 0, Color = "000000" },
            new RightBorder { Val = BorderValues.None, Size = 0, Color = "000000" },
            new InsideHorizontalBorder { Val = BorderValues.None, Size = 0, Color = "000000" },
            new InsideVerticalBorder { Val = BorderValues.None, Size = 0, Color = "000000" }));
    table.Append(tblPr);

    var grid = new TableGrid();
    foreach (var width in scaledWidths)
    {
        grid.Append(new GridColumn { Width = width.ToString() });
    }
    table.Append(grid);

    var headerRow = new TableRow(new TableRowProperties(new TableHeader()));
    for (var i = 0; i < headers.Length; i++)
    {
        headerRow.Append(CreateCell(headers[i], scaledWidths[i], true, true));
    }
    table.Append(headerRow);

    foreach (var row in rows)
    {
        var tr = new TableRow();
        for (var i = 0; i < row.Length; i++)
        {
            tr.Append(CreateCell(row[i], scaledWidths[i], false, false));
        }
        table.Append(tr);
    }

    body.Append(table);
    body.Append(new Paragraph(new ParagraphProperties(
        new SpacingBetweenLines { Before = "0", After = "220", Line = "360", LineRule = LineSpacingRuleValues.Auto })));
}

static TableCell CreateCell(string text, int width, bool bold, bool addBottomBorder)
{
    var tcPr = new TableCellProperties(
        new TableCellWidth { Width = width.ToString(), Type = TableWidthUnitValues.Dxa },
        new TableCellVerticalAlignment { Val = TableVerticalAlignmentValues.Center });

    if (addBottomBorder)
    {
        tcPr.Append(new TableCellBorders(
            new BottomBorder { Val = BorderValues.Single, Size = 8, Color = "000000" }));
    }

    var paragraph = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "0", After = "0", Line = "280", LineRule = LineSpacingRuleValues.Auto }),
        new Run(
            CreateTableRunProperties(bold),
            new Text(text) { Space = SpaceProcessingModeValues.Preserve }));

    return new TableCell(tcPr, paragraph);
}

static RunProperties CreateTableRunProperties(bool bold)
{
    var props = new RunProperties(
        new RunFonts
        {
            Ascii = "Times New Roman",
            HighAnsi = "Times New Roman",
            EastAsia = "SimSun",
            ComplexScript = "Times New Roman"
        },
        new FontSize { Val = "21" },
        new FontSizeComplexScript { Val = "21" });

    if (bold)
    {
        props.Append(new Bold());
        props.Append(new BoldComplexScript());
    }

    return props;
}

static void AppendImage(Body body, MainDocumentPart mainPart, string imagePath, double widthInches, int pixelWidth, int pixelHeight)
{
    using var stream = File.OpenRead(imagePath);
    var imagePart = mainPart.AddImagePart(ImagePartType.Png);
    imagePart.FeedData(stream);

    const int emusPerInch = 914400;
    var widthEmus = (long)(widthInches * emusPerInch);
    var heightEmus = (long)(pixelHeight * (widthEmus / (double)pixelWidth));

    var relationshipId = mainPart.GetIdOfPart(imagePart);
    var element =
        new Drawing(
            new DW.Inline(
                new DW.Extent { Cx = widthEmus, Cy = heightEmus },
                new DW.EffectExtent
                {
                    LeftEdge = 0L,
                    TopEdge = 0L,
                    RightEdge = 0L,
                    BottomEdge = 0L
                },
                new DW.DocProperties { Id = (UInt32Value)1U, Name = Path.GetFileName(imagePath) },
                new DW.NonVisualGraphicFrameDrawingProperties(
                    new A.GraphicFrameLocks { NoChangeAspect = true }),
                new A.Graphic(
                    new A.GraphicData(
                        new PIC.Picture(
                            new PIC.NonVisualPictureProperties(
                                new PIC.NonVisualDrawingProperties { Id = (UInt32Value)0U, Name = Path.GetFileName(imagePath) },
                                new PIC.NonVisualPictureDrawingProperties()),
                            new PIC.BlipFill(
                                new A.Blip { Embed = relationshipId },
                                new A.Stretch(new A.FillRectangle())),
                            new PIC.ShapeProperties(
                                new A.Transform2D(
                                    new A.Offset { X = 0L, Y = 0L },
                                    new A.Extents { Cx = widthEmus, Cy = heightEmus }),
                                new A.PresetGeometry(new A.AdjustValueList()) { Preset = A.ShapeTypeValues.Rectangle })))
                    { Uri = "http://schemas.openxmlformats.org/drawingml/2006/picture" }))
            {
                DistanceFromTop = 0U,
                DistanceFromBottom = 0U,
                DistanceFromLeft = 0U,
                DistanceFromRight = 0U
            });

    var paragraph = new Paragraph(
        new ParagraphProperties(
            new Justification { Val = JustificationValues.Center },
            new SpacingBetweenLines { Before = "120", After = "0", Line = "360", LineRule = LineSpacingRuleValues.Auto }),
        new Run(element));
    body.Append(paragraph);
}
