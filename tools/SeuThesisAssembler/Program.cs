using System.Text.RegularExpressions;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;

if (args.Length != 3)
{
    Console.Error.WriteLine("Usage: SeuThesisAssembler <template.docx> <source.docx> <output.docx>");
    return 1;
}

var templatePath = Path.GetFullPath(args[0]);
var sourcePath = Path.GetFullPath(args[1]);
var outputPath = Path.GetFullPath(args[2]);

if (!File.Exists(templatePath))
{
    Console.Error.WriteLine($"Template not found: {templatePath}");
    return 1;
}

if (!File.Exists(sourcePath))
{
    Console.Error.WriteLine($"Source not found: {sourcePath}");
    return 1;
}

File.Copy(templatePath, outputPath, overwrite: true);

using var outputDoc = WordprocessingDocument.Open(outputPath, true);
using var sourceDoc = WordprocessingDocument.Open(sourcePath, false);

var outputMain = outputDoc.MainDocumentPart ?? throw new InvalidOperationException("Output missing MainDocumentPart.");
var sourceMain = sourceDoc.MainDocumentPart ?? throw new InvalidOperationException("Source missing MainDocumentPart.");
var outputBody = outputMain.Document.Body ?? throw new InvalidOperationException("Output missing body.");
var sourceBody = sourceMain.Document.Body ?? throw new InvalidOperationException("Source missing body.");

FillCover(outputBody, ExtractTitle(sourceBody));
FillAiForm(outputBody, ExtractTitle(sourceBody));
ReplaceFrontMatter(outputBody, sourceBody);
ReplaceBodyReferencesAndAcknowledgements(outputBody, sourceBody);
EnsurePageBreaks(outputBody);
EnableUpdateFieldsOnOpen(outputMain);
outputMain.Document.Save();

Console.WriteLine($"Generated: {outputPath}");
return 0;

static void FillCover(Body body, string title)
{
    var coverTable = body.Descendants<Table>()
        .FirstOrDefault(t => NormalizeText(t).Contains("学号:") && NormalizeText(t).Contains("指导教师:"));

    if (coverTable is null)
    {
        return;
    }

    var rows = coverTable.Elements<TableRow>().ToList();
    if (rows.Count > 0)
    {
        var firstCell = rows[0].Elements<TableCell>().FirstOrDefault();
        if (firstCell is not null)
        {
            SetCellText(firstCell, title);
        }
    }

    if (rows.Count > 1)
    {
        var secondRowCell = rows[1].Elements<TableCell>().FirstOrDefault();
        if (secondRowCell is not null)
        {
            SetCellText(secondRowCell, string.Empty);
        }
    }

    foreach (var row in rows)
    {
        var cells = row.Elements<TableCell>().ToList();
        if (cells.Count < 2)
        {
            continue;
        }

        var label = NormalizeText(cells[0]);
        if (label.Contains("学号:"))
        {
            SetCellText(cells[1], "[待填写]");
        }
        else if (label.Contains("姓名:"))
        {
            SetCellText(cells[1], "朱世豪");
        }
        else if (label.Contains("学院:"))
        {
            SetCellText(cells[1], "[待填写]");
        }
        else if (label.Contains("专业:"))
        {
            SetCellText(cells[1], "[待填写]");
        }
        else if (label.Contains("指导教师:"))
        {
            SetCellText(cells[1], "赵瑞杰");
        }
        else if (label.Contains("起止日期:"))
        {
            SetCellText(cells[1], "[待填写]");
        }
    }
}

static void FillAiForm(Body body, string title)
{
    var aiTable = body.Descendants<Table>()
        .FirstOrDefault(t => NormalizeText(t).Contains("是否使用生成式人工智能") && NormalizeText(t).Contains("工具、版本号"));

    if (aiTable is null)
    {
        return;
    }

    var rows = aiTable.Elements<TableRow>().ToList();
    if (rows.Count >= 5)
    {
        var row0 = rows[0].Elements<TableCell>().ToList();
        if (row0.Count >= 2)
        {
            SetCellText(row0[1], title);
        }

        var row1 = rows[1].Elements<TableCell>().ToList();
        if (row1.Count >= 4)
        {
            SetCellText(row1[1], "[待填写]");
            SetCellText(row1[3], "朱世豪");
        }

        var row2 = rows[2].Elements<TableCell>().ToList();
        if (row2.Count >= 2)
        {
            SetCellText(row2[1], "■ 是    □ 否");
        }

        var row4 = rows[4].Elements<TableCell>().ToList();
        if (row4.Count >= 4)
        {
            SetCellText(row4[1], "OpenAI Codex（GPT-5）");
            SetCellText(row4[2], "论文文本生成与修改，代码调试，Word 排版辅助。");
            SetCellText(row4[3], "摘要、正文第1-6章、参考文献与致谢（以最终页码为准）");
        }

        return;
    }

    foreach (var row in rows)
    {
        var cells = row.Elements<TableCell>().ToList();
        if (cells.Count == 0)
        {
            continue;
        }

        var rowText = NormalizeText(row);
        if (rowText.Contains("课题名称") && cells.Count >= 2)
        {
            SetCellText(cells[1], title);
        }
    }
}

static void ReplaceFrontMatter(Body outputBody, Body sourceBody)
{
    var outputChildren = outputBody.Elements<OpenXmlElement>().ToList();
    var sourceChildren = sourceBody.Elements<OpenXmlElement>().ToList();

    var outCnAbs = FindIndex(outputChildren, t => t == "摘要");
    var outEnAbs = FindIndex(outputChildren, t => t == "ABSTRACT");
    var outToc = FindIndex(outputChildren, t => t == "目录");

    var srcCnAbs = FindIndex(sourceChildren, t => t == "摘要");
    var srcCnKeywords = FindIndex(sourceChildren, t => t.StartsWith("关键词:", StringComparison.Ordinal));
    var srcEnAbs = FindIndex(sourceChildren, t => t == "ABSTRACT");
    var srcEnKeywords = FindIndex(sourceChildren, t => t.StartsWith("KEYWORDS:", StringComparison.Ordinal) || t.StartsWith("KEYWORDS", StringComparison.Ordinal));

    if (outCnAbs >= 0 && outEnAbs > outCnAbs && srcCnAbs >= 0 && srcCnKeywords > srcCnAbs)
    {
        ReplaceRange(outputBody, outCnAbs + 1, outEnAbs - 1, CloneRange(sourceChildren, srcCnAbs + 1, srcCnKeywords));
    }

    outputChildren = outputBody.Elements<OpenXmlElement>().ToList();
    outEnAbs = FindIndex(outputChildren, t => t == "ABSTRACT");
    outToc = FindIndex(outputChildren, t => t == "目录");

    if (outEnAbs >= 0 && outToc > outEnAbs && srcEnAbs >= 0 && srcEnKeywords > srcEnAbs)
    {
        ReplaceRange(outputBody, outEnAbs + 1, outToc - 1, CloneRange(sourceChildren, srcEnAbs + 1, srcEnKeywords));
    }
}

static void ReplaceBodyReferencesAndAcknowledgements(Body outputBody, Body sourceBody)
{
    var outputChildren = outputBody.Elements<OpenXmlElement>().ToList();
    var sourceChildren = sourceBody.Elements<OpenXmlElement>().ToList();

    var outBodyStart = FindIndex(outputChildren, t => t.StartsWith("第1章", StringComparison.Ordinal));
    var outRefs = FindIndex(outputChildren, t => t == "参考文献");
    var outThanks = FindIndex(outputChildren, t => t == "致谢");

    var srcBodyStart = FindIndex(sourceChildren, t => t.StartsWith("第1章", StringComparison.Ordinal));
    var srcRefs = FindIndex(sourceChildren, t => t == "参考文献");
    var srcThanks = FindIndex(sourceChildren, t => t == "致谢");

    if (outBodyStart >= 0 && outRefs > outBodyStart && srcBodyStart >= 0 && srcRefs > srcBodyStart)
    {
        ReplaceRange(outputBody, outBodyStart, outRefs - 1, CloneRange(sourceChildren, srcBodyStart, srcRefs - 1));
    }

    outputChildren = outputBody.Elements<OpenXmlElement>().ToList();
    outRefs = FindIndex(outputChildren, t => t == "参考文献");
    outThanks = FindIndex(outputChildren, t => t == "致谢");

    if (outRefs >= 0 && outThanks > outRefs && srcRefs >= 0 && srcThanks > srcRefs)
    {
        ReplaceRange(outputBody, outRefs + 1, outThanks - 1, CloneRange(sourceChildren, srcRefs + 1, srcThanks - 1));
    }

    outputChildren = outputBody.Elements<OpenXmlElement>().ToList();
    outThanks = FindIndex(outputChildren, t => t == "致谢");

    if (outThanks >= 0 && srcThanks >= 0)
    {
        var outLastContent = outputChildren.FindLastIndex(e => e is not SectionProperties);
        var srcLastContent = sourceChildren.FindLastIndex(e => e is not SectionProperties);
        ReplaceRange(outputBody, outThanks + 1, outLastContent, CloneRange(sourceChildren, srcThanks + 1, srcLastContent));
    }
}

static void EnsurePageBreaks(Body body)
{
    var paragraphs = body.Elements<Paragraph>().ToList();
    var headingParagraphs = paragraphs.Where(p =>
    {
        var text = NormalizeText(p);
        return Regex.IsMatch(text, @"^第[0-9一二三四五六七八九十]+章") || text == "参考文献" || text == "致谢";
    }).ToList();

    for (var i = 1; i < headingParagraphs.Count; i++)
    {
        var paragraph = headingParagraphs[i];
        if (HasImmediatePageBreakBefore(paragraph))
        {
            continue;
        }

        paragraph.InsertBeforeSelf(new Paragraph(new Run(new Break { Type = BreakValues.Page })));
    }
}

static bool HasImmediatePageBreakBefore(Paragraph paragraph)
{
    var previous = paragraph.PreviousSibling<Paragraph>();
    if (previous is null)
    {
        return false;
    }

    return previous.Descendants<Break>().Any(b => b.Type?.Value == BreakValues.Page);
}

static void EnableUpdateFieldsOnOpen(MainDocumentPart mainPart)
{
    var settingsPart = mainPart.DocumentSettingsPart ?? mainPart.AddNewPart<DocumentSettingsPart>();
    settingsPart.Settings ??= new Settings();

    var existing = settingsPart.Settings.GetFirstChild<UpdateFieldsOnOpen>();
    if (existing is null)
    {
        settingsPart.Settings.Append(new UpdateFieldsOnOpen { Val = true });
    }
    else
    {
        existing.Val = true;
    }

    settingsPart.Settings.Save();
}

static List<OpenXmlElement> CloneRange(List<OpenXmlElement> elements, int startInclusive, int endInclusive)
{
    var list = new List<OpenXmlElement>();
    for (var i = startInclusive; i <= endInclusive; i++)
    {
        if (i < 0 || i >= elements.Count)
        {
            continue;
        }

        if (elements[i] is SectionProperties)
        {
            continue;
        }

        list.Add(elements[i].CloneNode(true));
    }

    return list;
}

static void ReplaceRange(Body body, int startInclusive, int endInclusive, List<OpenXmlElement> replacements)
{
    var children = body.Elements<OpenXmlElement>().ToList();
    if (startInclusive < 0 || startInclusive >= children.Count)
    {
        return;
    }

    if (endInclusive < startInclusive)
    {
        var anchor = children[startInclusive];
        foreach (var element in replacements)
        {
            anchor.InsertBeforeSelf(element);
        }

        return;
    }

    endInclusive = Math.Min(endInclusive, children.Count - 1);
    var start = children[startInclusive];

    foreach (var element in replacements)
    {
        start.InsertBeforeSelf(element);
    }

    for (var i = startInclusive; i <= endInclusive; i++)
    {
        children[i].Remove();
    }
}

static int FindIndex(List<OpenXmlElement> elements, Func<string, bool> predicate)
{
    for (var i = 0; i < elements.Count; i++)
    {
        if (predicate(NormalizeText(elements[i])))
        {
            return i;
        }
    }

    return -1;
}

static string ExtractTitle(Body body)
{
    foreach (var paragraph in body.Elements<Paragraph>())
    {
        var text = GetText(paragraph).Trim();
        if (!string.IsNullOrWhiteSpace(text) && !text.StartsWith("Table of Contents", StringComparison.OrdinalIgnoreCase))
        {
            return text;
        }
    }

    return "毕业设计论文题目";
}

static string NormalizeText(OpenXmlElement element)
{
    var text = GetText(element);
    text = text.Replace(" ", string.Empty)
        .Replace("\r", string.Empty)
        .Replace("\n", string.Empty)
        .Replace("\t", string.Empty)
        .Replace("：", ":");
    return text;
}

static string GetText(OpenXmlElement element)
{
    return string.Concat(element.Descendants<Text>().Select(t => t.Text));
}

static void SetCellText(TableCell cell, string text)
{
    var firstParagraph = cell.Elements<Paragraph>().FirstOrDefault() ?? new Paragraph();
    var newParagraph = new Paragraph();

    if (firstParagraph.ParagraphProperties is not null)
    {
        newParagraph.ParagraphProperties = (ParagraphProperties)firstParagraph.ParagraphProperties.CloneNode(true);
    }

    var firstRunProps = firstParagraph.Elements<Run>().FirstOrDefault()?.RunProperties;
    var run = new Run();
    if (firstRunProps is not null)
    {
        run.RunProperties = (RunProperties)firstRunProps.CloneNode(true);
    }
    run.Append(new Text(text) { Space = SpaceProcessingModeValues.Preserve });
    newParagraph.Append(run);

    cell.RemoveAllChildren<Paragraph>();
    cell.Append(newParagraph);
}
