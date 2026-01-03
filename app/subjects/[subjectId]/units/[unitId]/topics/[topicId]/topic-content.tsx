"use client"

import { useState } from "react"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ChevronRight, Eye, EyeOff, BookOpen, Code, AlertCircle, ExternalLink, ChevronDown } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { subjectsData } from "@/lib/subjects-data"

export default function TopicContent({
    subjectId,
    unitId,
    topicId,
}: {
    subjectId: string
    unitId: string
    topicId: string
}) {
    const [examMode, setExamMode] = useState(false)
    const [showAnswers, setShowAnswers] = useState(!examMode)

    const subject = subjectsData[subjectId]
    const unit = subject?.units.find((u) => u.id === unitId)
    const topic = unit?.topics.find((t) => t.id === topicId)

    if (!subject || !unit || !topic) {
        notFound()
    }

    // Handle exam mode toggle
    const handleExamModeToggle = (checked: boolean) => {
        setExamMode(checked)
        setShowAnswers(!checked)
    }

    return (
        <div className="min-h-screen bg-background">
            <div className="container mx-auto px-4 py-8">
                <div className="flex flex-col lg:flex-row gap-8">
                    {/* Main Content */}
                    <main className="flex-1 max-w-4xl">
                        {/* Breadcrumb */}
                        <nav className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground mb-6">
                            <Link href="/" className="hover:text-foreground">
                                Home
                            </Link>
                            <ChevronRight className="h-4 w-4" />
                            <Link href="/subjects" className="hover:text-foreground">
                                Subjects
                            </Link>
                            <ChevronRight className="h-4 w-4" />
                            <Link href={`/subjects/${subjectId}`} className="hover:text-foreground">
                                {subject.shortTitle}
                            </Link>
                            <ChevronRight className="h-4 w-4" />
                            <span className="text-foreground">{topic.title}</span>
                        </nav>

                        {/* Topic Header */}
                        <div className="mb-8">
                            <Badge className="mb-3">{unit.title}</Badge>
                            <h1 className="text-4xl font-bold mb-4 text-balance">{topic.title}</h1>
                        </div>

                        {/* Exam Mode Toggle */}
                        <Card className="mb-8 bg-muted/50">
                            <CardContent className="flex items-center justify-between p-4">
                                <div className="flex items-center gap-2">
                                    {examMode ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                                    <div>
                                        <Label htmlFor="exam-mode" className="font-semibold cursor-pointer">
                                            Exam Mode
                                        </Label>
                                        <p className="text-sm text-muted-foreground">
                                            {examMode ? "Answers hidden for practice" : "All answers visible"}
                                        </p>
                                    </div>
                                </div>
                                <Switch id="exam-mode" checked={examMode} onCheckedChange={handleExamModeToggle} />
                            </CardContent>
                        </Card>

                        {/* Concept Explanation */}
                        <section className="mb-12">
                            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                                <BookOpen className="h-6 w-6" />
                                Concept Explanation
                            </h2>
                            <Card>
                                <CardContent className="prose prose-neutral dark:prose-invert max-w-none p-6">
                                    {topic.content.explanation.map((para, i) => (
                                        <p key={i} className="leading-relaxed mb-4 last:mb-0">
                                            {para}
                                        </p>
                                    ))}
                                </CardContent>
                            </Card>
                        </section>

                        {/* Key Points */}
                        {topic.content.keyPoints && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4">Key Points</h2>
                                <Card>
                                    <CardContent className="p-6">
                                        <ul className="space-y-2">
                                            {topic.content.keyPoints.map((point, i) => (
                                                <li key={i} className="flex gap-3">
                                                    <span className="text-primary mt-1">•</span>
                                                    <span className="leading-relaxed">{point}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            </section>
                        )}

                        {/* Diagrams */}
                        {topic.content.diagrams && topic.content.diagrams.length > 0 && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4">Diagrams</h2>
                                <div className="space-y-4">
                                    {topic.content.diagrams.map((diagram, i) => (
                                        <Collapsible key={i} defaultOpen>
                                            <Card>
                                                <CardHeader>
                                                    <CollapsibleTrigger className="flex items-center justify-between w-full group">
                                                        <CardTitle className="text-lg">{diagram.title}</CardTitle>
                                                        <ChevronDown className="h-5 w-5 transition-transform group-data-[state=open]:rotate-180" />
                                                    </CollapsibleTrigger>
                                                </CardHeader>
                                                <CollapsibleContent>
                                                    <CardContent>
                                                        {diagram.imageUrl ? (
                                                            <div className="rounded-lg overflow-hidden">
                                                                <img
                                                                    src={diagram.imageUrl}
                                                                    alt={diagram.title}
                                                                    className="w-full h-auto max-h-[500px] object-contain bg-white rounded-lg"
                                                                />
                                                                <p className="text-sm text-muted-foreground mt-3 text-center italic">{diagram.description}</p>
                                                            </div>
                                                        ) : (
                                                            <div className="bg-muted rounded-lg p-8 flex items-center justify-center min-h-[300px] border-2 border-dashed border-border">
                                                                <div className="text-center text-muted-foreground">
                                                                    <Code className="h-12 w-12 mx-auto mb-3 opacity-50" />
                                                                    <p className="font-medium">{diagram.title}</p>
                                                                    <p className="text-sm mt-1">{diagram.description}</p>
                                                                </div>
                                                            </div>
                                                        )}
                                                    </CardContent>
                                                </CollapsibleContent>
                                            </Card>
                                        </Collapsible>
                                    ))}
                                </div>
                            </section>
                        )}

                        {/* Worked Examples */}
                        {topic.content.examples && topic.content.examples.length > 0 && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4">Worked Examples</h2>
                                <div className="space-y-6">
                                    {topic.content.examples.map((example, i) => (
                                        <Card key={i}>
                                            <CardHeader>
                                                <CardTitle className="text-lg">
                                                    Example {i + 1}: {example.title}
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent className="space-y-4">
                                                <div>
                                                    <h4 className="font-semibold mb-2">Problem:</h4>
                                                    <p className="leading-relaxed text-muted-foreground">{example.problem}</p>
                                                </div>
                                                <Collapsible open={showAnswers}>
                                                    <CollapsibleTrigger asChild>
                                                        <Button variant="outline" size="sm" className="w-full bg-transparent">
                                                            {showAnswers ? "Hide Solution" : "Show Solution"}
                                                        </Button>
                                                    </CollapsibleTrigger>
                                                    <CollapsibleContent className="mt-4">
                                                        <div className="bg-muted p-4 rounded-lg">
                                                            <h4 className="font-semibold mb-2">Solution:</h4>
                                                            <div className="leading-relaxed space-y-2">
                                                                {example.solution.split("\n").map((line, idx) => (
                                                                    <p key={idx}>{line}</p>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    </CollapsibleContent>
                                                </Collapsible>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </section>
                        )}

                        {/* Practice Problems */}
                        {topic.content.problems && topic.content.problems.length > 0 && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4">Practice Problems</h2>
                                <div className="space-y-6">
                                    {topic.content.problems.map((problem, i) => (
                                        <Card key={i}>
                                            <CardHeader>
                                                <CardTitle className="text-lg">Problem {i + 1}</CardTitle>
                                            </CardHeader>
                                            <CardContent className="space-y-4">
                                                <p className="leading-relaxed">{problem.question}</p>
                                                <Collapsible open={showAnswers}>
                                                    <CollapsibleTrigger asChild>
                                                        <Button variant="outline" size="sm" className="w-full bg-transparent">
                                                            {showAnswers ? "Hide Answer" : "Show Answer"}
                                                        </Button>
                                                    </CollapsibleTrigger>
                                                    <CollapsibleContent className="mt-4">
                                                        <div className="bg-muted p-4 rounded-lg">
                                                            <h4 className="font-semibold mb-2">Answer:</h4>
                                                            <p className="leading-relaxed">{problem.answer}</p>
                                                        </div>
                                                    </CollapsibleContent>
                                                </Collapsible>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            </section>
                        )}

                        {/* Exam Tips */}
                        {topic.content.examTips && topic.content.examTips.length > 0 && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                                    <AlertCircle className="h-6 w-6" />
                                    Exam Tips & Important Notes
                                </h2>
                                <Card className="border-primary/50 bg-primary/5">
                                    <CardContent className="p-6">
                                        <ul className="space-y-3">
                                            {topic.content.examTips.map((tip, i) => (
                                                <li key={i} className="flex gap-3">
                                                    <span className="text-primary font-bold mt-1">→</span>
                                                    <span className="leading-relaxed">{tip}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            </section>
                        )}

                        {/* Extra Resources */}
                        {topic.content.resources && topic.content.resources.length > 0 && (
                            <section className="mb-12">
                                <h2 className="text-2xl font-bold mb-4">Additional Resources</h2>
                                <Card>
                                    <CardContent className="p-6">
                                        <ul className="space-y-3">
                                            {topic.content.resources.map((resource, i) => (
                                                <li key={i}>
                                                    <a
                                                        href={resource.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="flex items-center gap-2 text-primary hover:underline"
                                                    >
                                                        <ExternalLink className="h-4 w-4" />
                                                        {resource.title}
                                                    </a>
                                                    <p className="text-sm text-muted-foreground ml-6 mt-1">{resource.description}</p>
                                                </li>
                                            ))}
                                        </ul>
                                    </CardContent>
                                </Card>
                            </section>
                        )}
                    </main>

                    {/* Table of Contents Sidebar */}
                    <aside className="lg:w-64 shrink-0">
                        <div className="sticky top-8">
                            <Card>
                                <CardHeader>
                                    <CardTitle className="text-lg">On This Page</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <nav className="space-y-2 text-sm">
                                        <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                            Concept Explanation
                                        </a>
                                        {topic.content.keyPoints && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Key Points
                                            </a>
                                        )}
                                        {topic.content.diagrams && topic.content.diagrams.length > 0 && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Diagrams
                                            </a>
                                        )}
                                        {topic.content.examples && topic.content.examples.length > 0 && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Worked Examples
                                            </a>
                                        )}
                                        {topic.content.problems && topic.content.problems.length > 0 && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Practice Problems
                                            </a>
                                        )}
                                        {topic.content.examTips && topic.content.examTips.length > 0 && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Exam Tips
                                            </a>
                                        )}
                                        {topic.content.resources && topic.content.resources.length > 0 && (
                                            <a href="#" className="block py-1 text-muted-foreground hover:text-foreground">
                                                Resources
                                            </a>
                                        )}
                                    </nav>
                                </CardContent>
                            </Card>
                        </div>
                    </aside>
                </div>
            </div>
        </div>
    )
}
